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
#include "../../algorithms/public/ChromaFilterBank.hpp"
#include "../../algorithms/public/DCT.hpp"
#include "../../algorithms/public/Loudness.hpp"
#include "../../algorithms/public/MelBands.hpp"
#include "../../algorithms/public/NoveltyFeature.hpp"
#include "../../algorithms/public/STFT.hpp"
#include "../../algorithms/public/YINFFT.hpp"
#include "../../algorithms/util/TruePeak.hpp"
#include "../../data/TensorTypes.hpp"
#include <tuple>

namespace fluid {
namespace client {
namespace noveltyfeature {

enum NoveltyParamIndex { kFeature, kKernelSize, kFilterSize, kFFT };

constexpr auto NoveltyFeatureParams = defineParameters(
    EnumParam("algorithm", "Algorithm for Feature Extraction", 0, "Spectrum",
        "MFCC", "Chroma", "Pitch", "Loudness"),
    LongParamRuntimeMax<Primary>("kernelSize", "KernelSize", 3, Min(3), Odd()),
    LongParamRuntimeMax<Primary>(
        "filterSize", "Smoothing Filter Size", 1, Min(1)),
    FFTParam("fftSettings", "FFT Settings", 1024, -1, -1));

class NoveltyFeatureClient : public FluidBaseClient,
                             public AudioIn,
                             public ControlOut
{
public:
  using ParamDescType = decltype(NoveltyFeatureParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors()
  {
    return NoveltyFeatureParams;
  }

  NoveltyFeatureClient(ParamSetViewType& p, FluidContext& c)
      : mParams{p},
        mNovelty{get<kKernelSize>().max(), get<kFFT>().maxFrameSize(),
            get<kFilterSize>().max(), c.allocator()},
        mBufferedProcess{get<kFFT>().max(), 0, 1, 0,
            c.hostVectorSize(), c.allocator()},
        mSTFT{get<kFFT>().max(), get<kFFT>().max(), get<kFFT>().hopSize(), 0,
            c.allocator()},
        mSpectrum{get<kFFT>().maxFrameSize(), c.allocator()},
        mMagnitude{get<kFFT>().maxFrameSize(), c.allocator()},
        mBands{40, c.allocator()},
        mFeature{get<kFFT>().maxFrameSize(), c.allocator()},
        mMelBands{40, get<kFFT>().max(), c.allocator()},
        mDCT{40, 13, c.allocator()},
        mChroma{12, get<kFFT>().max(), c.allocator()},
        mYinFFT(get<kFFT>().maxFrameSize(), c.allocator()),
        mLoudness{get<kFFT>().max(), c.allocator()}
  {
    audioChannelsIn(1);
    controlChannelsOut({1, 1});
    setInputLabels({"audio input"});
    setOutputLabels({"estimated novelty"});
  }


  void initAlgorithms(index feature, index windowSize, FluidContext& c)
  {
    nDims = 2;
    if (feature < 4)
    {
      mSTFT.resize(
          get<kFFT>().winSize(), get<kFFT>().fftSize(), get<kFFT>().hopSize());
    }
    if (feature == 0) { nDims = get<kFFT>().frameSize(); }
    else if (feature == 1)
    {
      mMelBands.init(20, 20e3, 40, get<kFFT>().frameSize(), sampleRate(),
          get<kFFT>().winSize(), c.allocator());
      mDCT.init(40, 13, c.allocator());
      nDims = 13;
    }
    else if (feature == 2)
    {
      mChroma.init(
          12, get<kFFT>().frameSize(), 440, sampleRate(), c.allocator());
      nDims = 12;
    }
    else if (feature == 4)
    {
      mLoudness.init(windowSize, sampleRate(), c.allocator());
    }
    mNovelty.init(get<kKernelSize>(), get<kFilterSize>(), nDims, c.allocator());
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
      std::vector<HostVector<T>>& output, FluidContext& c)
  {
    using algorithm::NoveltyFeature;


    if (!input[0].data() || !output[0].data()) return;

    index hostVecSize = input[0].size();
    assert(hostVecSize == c.hostVectorSize());
    index windowSize = get<kFFT>().winSize();
    index frameSize = get<kFFT>().frameSize();
    index featureIdx = get<kFeature>();
    if (mParamsTracker.changed(hostVecSize, get<kFeature>(), get<kKernelSize>(),
            get<kFilterSize>(), windowSize, sampleRate()))
    {
      initAlgorithms(featureIdx, windowSize, c);
      mBufferedProcess = BufferedProcess{get<kFFT>().max(), 0, 1, 0,
            c.hostVectorSize(), c.allocator()};
    }

    auto spectrum = mSpectrum(Slice(0, frameSize));
    auto magnitude = mMagnitude(Slice(0, frameSize));
    auto feature = mFeature(Slice(0, nDims));

    mBufferedProcess.push(FluidTensorView<T, 2>(input[0]));
    mBufferedProcess.processInput(
        windowSize, get<kFFT>().hopSize(), c, [&, this](RealMatrixView in) {
          switch (featureIdx)
          {
          case 0:
            mSTFT.processFrame(in.row(0), spectrum);
            mSTFT.magnitude(spectrum, feature);
            break;
          case 1:
            mSTFT.processFrame(in.row(0), spectrum);
            mSTFT.magnitude(spectrum, magnitude);
            mMelBands.processFrame(
                magnitude, mBands, false, false, true, c.allocator());
            mDCT.processFrame(mBands, feature);
            break;
          case 2:
            mSTFT.processFrame(in.row(0), spectrum);
            mSTFT.magnitude(spectrum, magnitude);
            mChroma.processFrame(magnitude, feature, 20, 5000);
            break;
          case 3:
            mSTFT.processFrame(in.row(0), spectrum);
            mSTFT.magnitude(spectrum, magnitude);
            mYinFFT.processFrame(
                magnitude, feature, 20, 5000, sampleRate(), c.allocator());
            break;
          case 4:
            mLoudness.processFrame(
                in.row(0), feature, true, true, c.allocator());
            break;
          }
          mDescriptor = mNovelty.processFrame(feature, c.allocator());
        });

    output[0](0) = static_cast<T>(mDescriptor);
  }

  index latency() const
  {
    index filterSize = get<kFilterSize>();
    if (filterSize % 2) filterSize++;
    return get<kFFT>().hopSize() *
           (1 + ((get<kKernelSize>() + 1) >> 1) + (filterSize >> 1));
  }

  AnalysisSize analysisSettings()
  {
    return {get<kFFT>().winSize(), get<kFFT>().hopSize()};
  }

  void reset(FluidContext& c)
  {
    mBufferedProcess.reset();
    initAlgorithms(get<kFeature>(), get<kFFT>().winSize(), c);
  }

private:
  algorithm::NoveltyFeature mNovelty;
  ParameterTrackChanges<index, index, index, index, index, double>
                                       mParamsTracker;
  BufferedProcess                      mBufferedProcess;
  algorithm::STFT                      mSTFT;
  FluidTensor<std::complex<double>, 1> mSpectrum;
  FluidTensor<double, 1>               mMagnitude;
  FluidTensor<double, 1>               mBands;
  FluidTensor<double, 1>               mFeature;
  double                               mDescriptor;
  algorithm::MelBands                  mMelBands;
  algorithm::DCT                       mDCT;
  algorithm::ChromaFilterBank          mChroma;
  algorithm::YINFFT                    mYinFFT;
  algorithm::Loudness                  mLoudness;
  index                                nDims;
};
} // namespace noveltyfeature

using RTNoveltyFeatureClient =
    ClientWrapper<noveltyfeature::NoveltyFeatureClient>;

auto constexpr NRTNoveltyFeatureParams = makeNRTParams<RTNoveltyFeatureClient>(
    InputBufferParam("source", "Source Buffer"),
    BufferParam("features", "Feature Buffer"));

using NRTNoveltyFeatureClient =
    NRTControlAdaptor<noveltyfeature::NoveltyFeatureClient,
        decltype(NRTNoveltyFeatureParams), NRTNoveltyFeatureParams, 1, 1>;

using NRTThreadedNoveltyFeatureClient =
    NRTThreadingAdaptor<NRTNoveltyFeatureClient>;

} // namespace client
} // namespace fluid
