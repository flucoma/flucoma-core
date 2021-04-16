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
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/DCT.hpp"
#include "../../algorithms/public/Loudness.hpp"
#include "../../algorithms/public/MelBands.hpp"
#include "../../algorithms/public/NoveltySegmentation.hpp"
#include "../../algorithms/public/STFT.hpp"
#include "../../algorithms/public/YINFFT.hpp"
#include "../../algorithms/util/TruePeak.hpp"
#include "../../data/TensorTypes.hpp"
#include <tuple>

namespace fluid {
namespace client {
namespace noveltyslice {

enum NoveltyParamIndex {
  kFeature,
  kKernelSize,
  kThreshold,
  kFilterSize,
  kDebounce,
  kFFT,
  kMaxFFTSize,
  kMaxKernelSize,
  kMaxFilterSize,
};

constexpr auto NoveltySliceParams = defineParameters(
    EnumParam("feature", "Feature", 0, "Spectrum", "MFCC", "Pitch", "Loudness"),
    LongParam("kernelSize", "KernelSize", 3, Min(3), Odd(),
              UpperLimit<kMaxKernelSize>()),
    FloatParam("threshold", "Threshold", 0.5, Min(0)),
    LongParam("filterSize", "Smoothing Filter Size", 1, Min(1),
              UpperLimit<kMaxFilterSize>()),
    LongParam("minSliceLength", "Minimum Length of Slice", 2, Min(0)),
    FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4),
                           PowerOfTwo{}),
    LongParam<Fixed<true>>("maxKernelSize", "Maxiumm Kernel Size", 101, Min(3),
                           Odd()),
    LongParam<Fixed<true>>("maxFilterSize", "Maxiumm Filter Size", 100,
                           Min(1)));

class NoveltySliceClient : public FluidBaseClient,
                           public AudioIn,
                           public AudioOut
{
public:
  using ParamDescType = decltype(NoveltySliceParams);

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
    return NoveltySliceParams;
  }

  NoveltySliceClient(ParamSetViewType& p)
      : mParams{p}, mNovelty{get<kMaxKernelSize>(), get<kMaxFilterSize>()},
        mSTFT{get<kFFT>().winSize(), get<kFFT>().fftSize(),
              get<kFFT>().hopSize()},
        mMelBands{40, get<kMaxFFTSize>()}, mLoudness{get<kMaxFFTSize>()}
  {
    audioChannelsIn(1);
    audioChannelsOut(1);
  }


  void initAlgorithms(index feature, index windowSize)
  {
    index nDims = 2;
    if (feature < 3)
    {
      mSpectrum.resize(get<kFFT>().frameSize());
      mMagnitude.resize(get<kFFT>().frameSize());
      mSTFT = algorithm::STFT(get<kFFT>().winSize(), get<kFFT>().fftSize(),
                              get<kFFT>().hopSize());
    }
    if (feature == 0) { nDims = get<kFFT>().frameSize(); }
    else if (feature == 1)
    {
      mBands.resize(40);
      mMelBands.init(20, 2000, 40, get<kFFT>().frameSize(), sampleRate(),
                     get<kFFT>().winSize());
      mDCT.init(40, 13);
      nDims = 13;
    }
    else if (feature == 3)
    {
      mLoudness.init(windowSize, sampleRate());
    }
    mFeature.resize(nDims);
    mNovelty.init(get<kKernelSize>(), get<kFilterSize>(), nDims);
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    using algorithm::NoveltySegmentation;


    if (!input[0].data() || !output[0].data()) return;

    index hostVecSize = input[0].size();
    index windowSize = get<kFFT>().winSize();
    index feature = get<kFeature>();
    if (mParamsTracker.changed(hostVecSize, get<kFeature>(), get<kKernelSize>(),
                               get<kFilterSize>(), windowSize, sampleRate()))
    {
      mBufferedProcess.hostSize(hostVecSize);
      mBufferedProcess.maxSize(windowSize, windowSize,
                               FluidBaseClient::audioChannelsIn(),
                               FluidBaseClient::audioChannelsOut());
      initAlgorithms(feature, windowSize);
    }
    RealMatrix in(1, hostVecSize);
    in.row(0) = input[0];
    RealMatrix out(1, hostVecSize);
    index      frameOffset = 0; // in case kHopSize < hostVecSize
    mBufferedProcess.push(RealMatrixView(in));
    mBufferedProcess.process(
        windowSize, windowSize, get<kFFT>().hopSize(), c,
        [&, this](RealMatrixView in, RealMatrixView) {
          switch (feature)
          {
          case 0:
            mSTFT.processFrame(in.row(0), mSpectrum);
            mSTFT.magnitude(mSpectrum, mFeature);
            break;
          case 1:
            mSTFT.processFrame(in.row(0), mSpectrum);
            mSTFT.magnitude(mSpectrum, mMagnitude);
            mMelBands.processFrame(mMagnitude, mBands, false, false, true);
            mDCT.processFrame(mBands, mFeature);
            break;
          case 2:
            mSTFT.processFrame(in.row(0), mSpectrum);
            mSTFT.magnitude(mSpectrum, mMagnitude);
            mYinFFT.processFrame(mMagnitude, mFeature, 20, 5000, sampleRate());
            break;
          case 3:
            mLoudness.processFrame(in.row(0), mFeature, true, true);
            break;
          }
          if (frameOffset < out.row(0).size())
            out.row(0)(frameOffset) = mNovelty.processFrame(
                mFeature, get<kThreshold>(), get<kDebounce>());
          frameOffset += get<kFFT>().hopSize();
        });
    output[0] = out.row(0);
  }

  index latency()
  {
    index filterSize =  get<kFilterSize>();
    if(filterSize % 2) filterSize++;
    return get<kFFT>().hopSize() * (
      1 + ((get<kKernelSize>() + 1) >> 1) + (filterSize >> 1)
    );
  }

  void reset()
  {
    mBufferedProcess.reset();
    initAlgorithms(get<kFeature>(), get<kFFT>().winSize());
  }

private:
  algorithm::NoveltySegmentation mNovelty;
  ParameterTrackChanges<index, index, index, index, index, double>
                                       mParamsTracker;
  BufferedProcess                      mBufferedProcess;
  algorithm::STFT                      mSTFT;
  FluidTensor<std::complex<double>, 1> mSpectrum;
  FluidTensor<double, 1>               mMagnitude;
  FluidTensor<double, 1>               mBands;
  FluidTensor<double, 1>               mFeature;
  algorithm::MelBands                  mMelBands;
  algorithm::DCT                       mDCT{40, 13};
  algorithm::YINFFT                    mYinFFT;
  algorithm::Loudness                  mLoudness;
};
} // namespace noveltyslice

using RTNoveltySliceClient = ClientWrapper<noveltyslice::NoveltySliceClient>;

auto constexpr NRTNoveltySliceParams = makeNRTParams<RTNoveltySliceClient>(
    InputBufferParam("source", "Source Buffer"),
    BufferParam("indices", "Indices Buffer"));

using NRTNoveltySliceClient = NRTSliceAdaptor<noveltyslice::NoveltySliceClient,
                                              decltype(NRTNoveltySliceParams),
                                              NRTNoveltySliceParams, 1, 1>;

using NRTThreadingNoveltySliceClient =
    NRTThreadingAdaptor<NRTNoveltySliceClient>;

} // namespace client
} // namespace fluid
