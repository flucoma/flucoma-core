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
#include "../../algorithms/public/MelBands.hpp"
#include "../../data/TensorTypes.hpp"


namespace fluid {
namespace client {
namespace melbands {

enum MFCCParamIndex {
  kNBands,
  kMinFreq,
  kMaxFreq,
  kNormalize,
  kScale,
  kFFT
};

constexpr auto MelBandsParams = defineParameters(
    LongParamRuntimeMax<Primary>("numBands", "Number of Bands", 40, Min(2)),
    FloatParam("minFreq", "Low Frequency Bound", 20, Min(0)),
    FloatParam("maxFreq", "High Frequency Bound", 20000, Min(0)),
    EnumParam("normalize", "Normalize", 1, "No", "Yes"),
    EnumParam("scale", "Amplitude Scale", 0, "Linear", "dB"),
    FFTParam("fftSettings", "FFT Settings", 1024, -1, -1));

class MelBandsClient : public FluidBaseClient, public AudioIn, public ControlOut
{

public:
  using ParamDescType = decltype(MelBandsParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return MelBandsParams; }

  MelBandsClient(ParamSetViewType& p, FluidContext& c)
      : mParams{p},
        mSTFTBufferedProcess(get<kFFT>(),1,0,c.hostVectorSize(),c.allocator()),
        mMelBands(get<kNBands>().max(), get<kFFT>().max(),c.allocator()),
        mMagnitude{get<kFFT>().maxFrameSize(), c.allocator()},
        mBands{get<kNBands>().max(), c.allocator()}
  {
    audioChannelsIn(1);
    controlChannelsOut({1,get<kNBands>(),get<kNBands>().max()});
    setInputLabels({"audio in"});
    setOutputLabels({"mel band energies"}); 
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    using std::size_t;

    if (!input[0].data() || !output[0].data()) return;
    assert(controlChannelsOut().size && "No control channels");
    assert(output[0].size() >= controlChannelsOut().size &&
           "Too few output channels");
           
    index nBands = get<kNBands>();
    index winSize = get<kFFT>().winSize();
    index frameSize = get<kFFT>().frameSize();
           
    if (mTracker.changed(winSize,frameSize, nBands,
                         get<kNormalize>(), get<kMinFreq>(),
                         get<kMaxFreq>(), sampleRate()))
    {
      mMelBands.init(get<kMinFreq>(), get<kMaxFreq>(), nBands,
                     frameSize, sampleRate(),winSize, c.allocator());
      controlChannelsOut({1, nBands});
    }
    
    if (mHostSizeTracker.changed(c.hostVectorSize()))
    {
      mSTFTBufferedProcess =    STFTBufferedProcess<false>(get<kFFT>(),1,0,c.hostVectorSize(),c.allocator()); 
    }
    
    auto mags = mMagnitude(Slice(0,frameSize));
    auto bands = mBands(Slice(0,nBands));
    
    mSTFTBufferedProcess.processInput(
        get<kFFT>(), input, c, [&](ComplexMatrixView in) {
          algorithm::STFT::magnitude(in.row(0), mags);
          mMelBands.processFrame(mags, bands, get<kNormalize>() == 1,
                                 false, get<kScale>() == 1, c.allocator());
        });
    // for (index i = 0; i < get<kNBands>(); ++i)
    //   output[asUnsigned(i)](0) = static_cast<T>(mBands(i));
    output[0](Slice(0,nBands)) <<= bands;
    output[0](Slice(nBands, get<kNBands>().max() - nBands)).fill(0);
  }

  index latency() const { return get<kFFT>().winSize(); }

  void reset(FluidContext& c)
  {
    mSTFTBufferedProcess.reset();
    mMelBands.init(get<kMinFreq>(), get<kMaxFreq>(), get<kNBands>(),
                   get<kFFT>().frameSize(), sampleRate(),
                   get<kFFT>().winSize(), c.allocator());
  }

  AnalysisSize analysisSettings()
  {
    return { get<kFFT>().winSize(), get<kFFT>().hopSize() }; 
  }


private:
  ParameterTrackChanges<index, index, index, index, double, double, double>
                                                     mTracker;
  ParameterTrackChanges<index> mHostSizeTracker;
  STFTBufferedProcess<false> mSTFTBufferedProcess;

  algorithm::MelBands    mMelBands;
  FluidTensor<double, 1> mMagnitude;
  FluidTensor<double, 1> mBands;
};
} // namespace melbands

using RTMelBandsClient = ClientWrapper<melbands::MelBandsClient>;

auto constexpr NRTMelBandsParams = makeNRTParams<melbands::MelBandsClient>(
    InputBufferParam("source", "Source Buffer"),
    BufferParam("features", "Output Buffer"));

using NRTMelBandsClient =
    NRTControlAdaptor<melbands::MelBandsClient, decltype(NRTMelBandsParams),
                      NRTMelBandsParams, 1, 1>;

using NRTThreadedMelBandsClient = NRTThreadingAdaptor<NRTMelBandsClient>;

} // namespace client
} // namespace fluid
