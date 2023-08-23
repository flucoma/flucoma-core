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
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/CepstrumF0.hpp"
#include "../../algorithms/public/HPS.hpp"
#include "../../algorithms/public/YINFFT.hpp"
#include "../../data/TensorTypes.hpp"
#include <tuple>

namespace fluid {
namespace client {
namespace pitch {

enum PitchParamIndex {
  kSelect,
  kAlgorithm,
  kMinFreq,
  kMaxFreq,
  kUnit,
  kFFT
};

constexpr auto PitchParams = defineParameters(
    ChoicesParam("select","Selection of Outputs","pitch","confidence"),
    EnumParam("algorithm", "Algorithm", 2, "Cepstrum",
              "Harmonic Product Spectrum", "YinFFT"),
    FloatParam("minFreq", "Low Frequency Bound", 20, Min(0), Max(10000),
               UpperLimit<kMaxFreq>()),
    FloatParam("maxFreq", "High Frequency Bound", 10000, Min(1), Max(20000),
               LowerLimit<kMinFreq>()),
    EnumParam("unit", "Frequency Unit", 0, "Hz", "MIDI"),
    FFTParam("fftSettings", "FFT Settings", 1024, -1, -1));

class PitchClient : public FluidBaseClient, public AudioIn, public ControlOut
{
  using size_t = std::size_t;
  using CepstrumF0 = algorithm::CepstrumF0;
  using HPS = algorithm::HPS;
  using YINFFT = algorithm::YINFFT;
  
  static constexpr index mMaxFeatures = 2;

  constexpr static std::array<double (*)(double), 2> setPitchUnits{
      [](double x) { return x; },
      [](double x) { return x == 0 ? -999 : (69 + (12 * log2(x / 440.0))); }};

public:
  using ParamDescType = decltype(PitchParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return PitchParams; }

  PitchClient(ParamSetViewType& p, FluidContext& c)
      : mParams(p), mSTFTBufferedProcess(get<kFFT>(), 1, 0, c.hostVectorSize(), c.allocator()),
        mCepstrumF0(get<kFFT>().maxFrameSize(), c.allocator()),
        mYinFFT(get<kFFT>().maxFrameSize(), c.allocator()),
        mMagnitude(get<kFFT>().maxFrameSize(), c.allocator()),
        mDescriptors(2, c.allocator())
  {
    audioChannelsIn(1);
    controlChannelsOut({1,mMaxFeatures});
    setInputLabels({"audio input"});
    setOutputLabels({"pitch (hz or MIDI), pitch confidence (0-1)"});
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    if (!input[0].data() || !output[0].data()) return;
    assert(controlChannelsOut().size && "No control channels");
    assert(output[0].size() >= controlChannelsOut().size &&
           "Too few output channels");

    if (mParamTracker.changed(get<kFFT>().frameSize(), sampleRate(), c.hostVectorSize()))
    {
      mCepstrumF0.init(get<kFFT>().frameSize(), c.allocator());
      mSTFTBufferedProcess = STFTBufferedProcess(get<kFFT>(), 1, 0, c.hostVectorSize(), c.allocator());
//      mMagnitude.resize(get<kFFT>().frameSize());
    }
    
    FluidTensorView<double, 1> mags = mMagnitude(Slice(0,get<kFFT>().frameSize()));
            
    mSTFTBufferedProcess.processInput(
        get<kFFT>(), input, c, [&](ComplexMatrixView in) {
          algorithm::STFT::magnitude(in.row(0), mags);
          switch (get<kAlgorithm>())
          {
          case 0:
            mCepstrumF0.processFrame(mags, mDescriptors, get<kMinFreq>(),
                                    get<kMaxFreq>(), sampleRate(),c.allocator());
            break;
          case 1:
            mHPS.processFrame(mags, mDescriptors, 4, get<kMinFreq>(),
                             get<kMaxFreq>(), sampleRate(), c.allocator());
            break;
          case 2:
            mYinFFT.processFrame(mags, mDescriptors, get<kMinFreq>(),
                                get<kMaxFreq>(), sampleRate(), c.allocator());
            break;
          }
        });
    // pitch
    
    auto selection = get<kSelect>();
    index numSelected = asSigned(selection.count());
    index numOuts = std::min<index>(mMaxFeatures,numSelected);
    controlChannelsOut({1,numOuts, mMaxFeatures});
    
    index i = 0;

    //pitch
    if (selection[0])
    {
      output[0](i++) =
          static_cast<T>(setPitchUnits[asUnsigned(get<kUnit>())](mDescriptors(0)));
    }

    // pitch confidence
    if(selection[1])
      output[0](i) = static_cast<T>(mDescriptors(1));
      
    output[0](Slice(numOuts,mMaxFeatures - numOuts)).fill(0);         
  }
  index latency() const { return get<kFFT>().winSize(); }

  AnalysisSize analysisSettings()
  {
    return { get<kFFT>().winSize(), get<kFFT>().hopSize() }; 
  }

  void reset(FluidContext& c)
  {
    mSTFTBufferedProcess.reset();
    mCepstrumF0.init(get<kFFT>().frameSize(), c.allocator());
//    mMagnitude.resize(get<kFFT>().frameSize());
  }

private:
  ParameterTrackChanges<index, double, index>        mParamTracker;
  
  STFTBufferedProcess<> mSTFTBufferedProcess;

  CepstrumF0             mCepstrumF0;
  HPS                    mHPS;
  YINFFT                 mYinFFT;
  FluidTensor<double, 1> mMagnitude;
  FluidTensor<double, 1> mDescriptors;
};
} // namespace pitch

using RTPitchClient = ClientWrapper<pitch::PitchClient>;

auto constexpr NRTPitchParams = makeNRTParams<pitch::PitchClient>(
    InputBufferParam("source", "Source Buffer"),
    BufferParam("features", "Features Buffer"));

using NRTPitchClient =
    NRTControlAdaptor<pitch::PitchClient, decltype(NRTPitchParams),
                      NRTPitchParams, 1, 1>;

using NRTThreadedPitchClient = NRTThreadingAdaptor<NRTPitchClient>;

} // namespace client
} // namespace fluid
