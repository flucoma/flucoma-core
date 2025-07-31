/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
#pragma once

#include "../common/AudioClient.hpp"
#include "../common/BufferedProcess.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTrackChanges.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/DCT.hpp"
#include "../../algorithms/public/MelBands.hpp"
#include "../../data/TensorTypes.hpp"

namespace fluid {
namespace client {
namespace mfcc {

enum MFCCParamIndex {
  kNCoefs,
  kNBands,
  kDrop0,
  kMinFreq,
  kMaxFreq,
  kFFT
};

constexpr auto MFCCParams = defineParameters(
    LongParamRuntimeMax<Primary>("numCoeffs", "Number of Cepstral Coefficients", 13,
              Min(2),
              UpperLimit<kNBands>()),
    LongParamRuntimeMax<Primary>("numBands", "Number of Bands", 40, Min(2),
              FrameSizeUpperLimit<kFFT>(), LowerLimit<kNCoefs>()),
    LongParam("startCoeff", "Output Coefficient Offset", 0, Min(0),
              Max(1)), // this needs to be programmatically changed to start+num
                       // coeffs <= numBands as discussed
    FloatParam("minFreq", "Low Frequency Bound", 20, Min(0)),
    FloatParam("maxFreq", "High Frequency Bound", 20000, Min(0)),
    FFTParam("fftSettings", "FFT Settings", 1024, -1, -1));

class MFCCClient : public FluidBaseClient, public AudioIn, public ControlOut
{
public:
  using ParamDescType = decltype(MFCCParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p)
  {
    mParams = p;
    controlChannelsOut({1, get<kNCoefs>(), get<kNCoefs>().max()});
  }

  template <size_t N>
  auto get() const -> decltype(mParams.get().template get<N>())
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return MFCCParams; }

  MFCCClient(ParamSetViewType& p, FluidContext& c)
      : mParams{p}, mSTFTBufferedProcess(get<kFFT>(), 1, 0, c.hostVectorSize(), c.allocator()),
        mMelBands(get<kFFT>().max(), get<kFFT>().max(), c.allocator()),
        mDCT(get<kFFT>().max(), get<kNCoefs>().max() + 1, c.allocator()),
        mMagnitude(get<kFFT>().maxFrameSize(), c.allocator()),
        mBands(get<kNBands>().max(), c.allocator()),
        mCoefficients(get<kNCoefs>().max() + 1, c.allocator())
  {
    audioChannelsIn(1);
    controlChannelsOut({1, get<kNCoefs>(), get<kNCoefs>().max()});
    setInputLabels({"audio input"});
    setOutputLabels({"MFCCs"});
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    using std::size_t;

    if (!input[0].data() || !output[0].data()) return;
    assert(controlChannelsOut().count && "No control channels");
    assert(output[0].size() >= controlChannelsOut().size &&
           "Too few output channels");

    bool has0 = !get<kDrop0>();
    index nCoefs = get<kNCoefs>();
    index nBands = get<kNBands>();
    index frameSize = get<kFFT>().frameSize();

    if (mTracker.changed(frameSize, nCoefs + !has0,
                         nBands, get<kMinFreq>(), get<kMaxFreq>(),
                         sampleRate()))
    {
      mMelBands.init(get<kMinFreq>(), get<kMaxFreq>(), nBands,
                     get<kFFT>().frameSize(), sampleRate(),
                     get<kFFT>().winSize(), c.allocator());
      mDCT.init(get<kNBands>(), std::min(nCoefs + !has0, nBands), c.allocator());
      controlChannelsOut({1, nCoefs});
    }

    if (mHostSizeTracker.changed(c.hostVectorSize()))
    {
      mSTFTBufferedProcess =    STFTBufferedProcess<false>(get<kFFT>(),1,0,c.hostVectorSize(),c.allocator());
    }

    auto mags  = mMagnitude(Slice(0,frameSize));
    auto bands = mBands(Slice(0,nBands));
    auto coefs = mCoefficients(Slice(0, std::min(nCoefs + !has0, nBands))); //making sure that we don't ask for more than nBands coeff in case of has0

    mSTFTBufferedProcess.processInput(
        get<kFFT>(), input, c, [&](ComplexMatrixView in) {
          algorithm::STFT::magnitude(in.row(0), mags);
          mMelBands.processFrame(mags, bands, false, false, true, c.allocator());
          mDCT.processFrame(bands, coefs);
        });
  
      output[0](Slice(0, nCoefs)) <<= mCoefficients(Slice(get<kDrop0>(), nCoefs)); // copying from has0 for nCoefs
      output[0](Slice(nCoefs, get<kNCoefs>().max() - nCoefs)).fill(0);
  }

  index latency() const { return get<kFFT>().winSize(); }

  void reset(FluidContext& c)
  {
    index nBands = get<kNBands>();

    mSTFTBufferedProcess.reset();
    mMagnitude.resize(get<kFFT>().frameSize());
    mBands.resize(nBands);
    mCoefficients.resize(get<kNCoefs>().max() + 1); //same as line 79
    mMelBands.init(get<kMinFreq>(), get<kMaxFreq>(), nBands,
                   get<kFFT>().frameSize(), sampleRate(),
                   get<kFFT>().winSize(), c.allocator());
    mDCT.init(nBands, std::min((get<kNCoefs>() + get<kDrop0>()), nBands)); //making sure that we don't ask for more than nBands coeff in case of has0
  }

  AnalysisSize analysisSettings()
  {
    return { get<kFFT>().winSize(), get<kFFT>().hopSize() }; 
  }


private:
  ParameterTrackChanges<index, index, index, double, double, double> mTracker;
  ParameterTrackChanges<index> mHostSizeTracker;
  
  STFTBufferedProcess<false> mSTFTBufferedProcess;

  algorithm::MelBands    mMelBands;
  algorithm::DCT         mDCT;
  FluidTensor<double, 1> mMagnitude;
  FluidTensor<double, 1> mBands;
  FluidTensor<double, 1> mCoefficients;
};
} // namespace mfcc

using RTMFCCClient = ClientWrapper<mfcc::MFCCClient>;

auto constexpr NRTMFCCParams =
    makeNRTParams<mfcc::MFCCClient>(InputBufferParam("source", "Source Buffer"),
                                    BufferParam("features", "Output Buffer"));

using NRTMFCCClient =
    NRTControlAdaptor<mfcc::MFCCClient, decltype(NRTMFCCParams), NRTMFCCParams,
                      1, 1>;

using NRTThreadedMFCCClient = NRTThreadingAdaptor<NRTMFCCClient>;

} // namespace client
} // namespace fluid
