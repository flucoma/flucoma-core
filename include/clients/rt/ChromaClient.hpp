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
#include "../../algorithms/public/ChromaFilterBank.hpp"
#include "../../data/TensorTypes.hpp"

namespace fluid {
namespace client {
namespace chroma {

enum ChromaParamIndex {
  kNChroma,
  kRef,
  kNorm,
  kMinFreq,
  kMaxFreq,
  kFFT
};

constexpr auto ChromaParams = defineParameters(
    LongParamRuntimeMax<Primary>("numChroma", "Number of Chroma Bins per Octave", 12, Min(2)),
    FloatParam("ref", "Reference frequency", 440, Min(0), Max(22000)),
    EnumParam("normalize", "Normalize Frame", 0, "None", "Sum", "Max"),
    FloatParam("minFreq", "Low Frequency Bound", 0, Min(0)),
    FloatParam("maxFreq", "High Frequency Bound", -1, Min(-1)),
    FFTParam("fftSettings", "FFT Settings", 1024, -1, -1));

class ChromaClient : public FluidBaseClient, public AudioIn, public ControlOut
{

public:
  using ParamDescType = decltype(ChromaParams);
  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return ChromaParams; }

  ChromaClient(ParamSetViewType& p, FluidContext const& c)
      : mParams{p},
        mSTFTBufferedProcess(get<kFFT>(), 1, 0, c.hostVectorSize(), c.allocator()),
        mAlgorithm(get<kNChroma>().max(), get<kFFT>().max(), c.allocator())
  {
    mMagnitude = FluidTensor<double, 1>(get<kFFT>().maxFrameSize(), c.allocator());
    mChroma = FluidTensor<double, 1>(get<kNChroma>().max(), c.allocator());
    audioChannelsIn(1);
    controlChannelsOut({1,get<kNChroma>(),get<kNChroma>().max()});
    setInputLabels({"audio in"});
    setOutputLabels({"energies at chroma bins"});
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
           
    index frameSize = get<kFFT>().frameSize();
    index nChroma = get<kNChroma>();
           
    if (mTracker.changed(frameSize, nChroma, get<kRef>(), sampleRate()))
    {
      mAlgorithm.init(nChroma, frameSize, get<kRef>(), sampleRate(), c.allocator());
      controlChannelsOut({1, nChroma});
    }
    
    if(mHostVSTracker.changed(c.hostVectorSize()))
        mSTFTBufferedProcess = STFTBufferedProcess<false>(get<kFFT>(), 1, 0, c.hostVectorSize(), c.allocator()); 
    
    auto mags = mMagnitude(Slice(0,frameSize));
    auto chroma = mChroma(Slice(0,nChroma));

    mSTFTBufferedProcess.processInput(
        get<kFFT>(), input, c, [&](ComplexMatrixView in) {
          algorithm::STFT::magnitude(in.row(0), mags);
          mAlgorithm.processFrame(mags, chroma, get<kMinFreq>(),
                                  get<kMaxFreq>(), get<kNorm>());
        });

    output[0](Slice(0,nChroma)) <<= chroma;
    output[0](Slice(nChroma, get<kNChroma>().max() - nChroma)).fill(0);
  }

  index latency() const { return get<kFFT>().winSize(); }

  void reset(FluidContext const& c)
  {
    mSTFTBufferedProcess.reset();
    mAlgorithm.init(get<kNChroma>(), get<kFFT>().frameSize(), get<kRef>(),
                    sampleRate(), c.allocator());
  }

  AnalysisSize analysisSettings()
  {
    return { get<kFFT>().winSize(), get<kFFT>().hopSize() }; 
  }

  private:
    ParameterTrackChanges<index, index, double, double> mTracker;
    ParameterTrackChanges<index> mHostVSTracker;
    STFTBufferedProcess<false>  mSTFTBufferedProcess;

    algorithm::ChromaFilterBank mAlgorithm;
    FluidTensor<double, 1>      mMagnitude;
    FluidTensor<double, 1>      mChroma;
  };
} // namespace chroma

using RTChromaClient = ClientWrapper<chroma::ChromaClient>;

auto constexpr NRTChromaParams = makeNRTParams<chroma::ChromaClient>(
    InputBufferParam("source", "Source Buffer"),
    BufferParam("features", "Output Buffer"));

using NRTChromaClient =
    NRTControlAdaptor<chroma::ChromaClient, decltype(NRTChromaParams),
                      NRTChromaParams, 1, 1>;

using NRTThreadedChromaClient = NRTThreadingAdaptor<NRTChromaClient>;

} // namespace client
} // namespace fluid
