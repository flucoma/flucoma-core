/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
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
  kMaxNChroma,
  kFFT,
  kMaxFFTSize
};

constexpr auto ChromaParams = defineParameters(
    LongParam("numChroma", "Number of Chroma Bins per Octave", 12, Min(2),
              UpperLimit<kMaxNChroma>()),
    FloatParam("ref", "Reference frequency", 440, Min(0), Max(22000)),
    EnumParam("normalize", "Normalize Frame", 0, "None", "Sum", "Max"),
    FloatParam("minFreq", "Low Frequency Bound", 0, Min(0)),
    FloatParam("maxFreq", "High Frequency Bound", -1, Min(-1)),
    LongParam<Fixed<true>>("maxNumChroma", "Maximum Number of Chroma Bins", 120, Min(2),
                           MaxFrameSizeUpperLimit<kMaxFFTSize>()),
    FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384));

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

  ChromaClient(ParamSetViewType& p)
      : mParams{p}, mSTFTBufferedProcess(get<kMaxFFTSize>(), 1, 0),
        mAlgorithm(get<kMaxNChroma>(), get<kMaxFFTSize>())
  {
    mChroma = FluidTensor<double, 1>(get<kNChroma>());
    audioChannelsIn(1);
    controlChannelsOut({1,get<kMaxNChroma>()});
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
    if (mTracker.changed(get<kFFT>().frameSize(), get<kNChroma>(), get<kRef>(),
                         sampleRate()))
    {
      mMagnitude.resize(get<kFFT>().frameSize());
      mChroma.resize(get<kNChroma>());
      mAlgorithm.init(get<kNChroma>(), get<kFFT>().frameSize(), get<kRef>(),
                      sampleRate());
    }

    mSTFTBufferedProcess.processInput(
        mParams, input, c, [&](ComplexMatrixView in) {
          algorithm::STFT::magnitude(in.row(0), mMagnitude);
          mAlgorithm.processFrame(mMagnitude, mChroma, get<kMinFreq>(),
                                  get<kMaxFreq>(), get<kNorm>());
        });

    output[0](Slice(0,get<kNChroma>())) = mChroma; 
    output[0](Slice(get<kNChroma>(), get<kMaxNChroma>() - get<kNChroma>())).fill(0); 
  }

  index latency() { return get<kFFT>().winSize(); }

  void reset()
  {
    mSTFTBufferedProcess.reset();
    mAlgorithm.init(get<kNChroma>(), get<kFFT>().frameSize(), get<kRef>(),
                    sampleRate());
  }

  index controlRate() { return get<kFFT>().hopSize(); }

private:
  ParameterTrackChanges<index, index, double, double> mTracker;
  STFTBufferedProcess<ParamSetViewType, kFFT, false>  mSTFTBufferedProcess;

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
