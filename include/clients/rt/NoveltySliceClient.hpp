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

extern auto constexpr NoveltyParams = defineParameters(
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

template <typename T>
class NoveltySliceClient
    : public FluidBaseClient<decltype(NoveltyParams), NoveltyParams>,
      public AudioIn,
      public AudioOut
{

  using HostVector = FluidTensorView<T, 1>;

public:
  NoveltySliceClient(ParamSetViewType& p) : FluidBaseClient(p)
  {
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::audioChannelsOut(1);
  }

  void process(std::vector<HostVector>& input, std::vector<HostVector>& output,
               FluidContext& c)
  {
    using algorithm::NoveltySegmentation;

    if (!input[0].data() || !output[0].data()) return;

    index hostVecSize = input[0].size();
    index windowSize = get<kFFT>().winSize();
    index feature = get<kFeature>();
    if (mParamsTracker.changed(hostVecSize, get<kFeature>(), get<kKernelSize>(),
                               get<kThreshold>(), get<kFilterSize>(),
                               windowSize, sampleRate()))
    {
      mBufferedProcess.hostSize(hostVecSize);
      mBufferedProcess.maxSize(windowSize, windowSize,
                               FluidBaseClient::audioChannelsIn(),
                               FluidBaseClient::audioChannelsOut());
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
                       true, false, get<kFFT>().winSize(), false);
        mDCT.init(40, 13);
        nDims = 13;
      }
      else if (feature == 3)
      {
        mLoudness.init(windowSize, sampleRate());
      }
      mFeature.resize(nDims);
      mNovelty.init(get<kKernelSize>(), get<kThreshold>(), get<kFilterSize>(),
                    nDims);
    }

    mNovelty.setMinSliceLength(get<kDebounce>());
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
            mMelBands.processFrame(mMagnitude, mBands);
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
            out.row(0)(frameOffset) = mNovelty.processFrame(mFeature);
          frameOffset += get<kFFT>().hopSize();
        });
    output[0] = out.row(0);
  }

  index latency()
  {
    return get<kFFT>().winSize() +
           (((get<kFilterSize>() / 2) + ((get<kKernelSize>() + 1) / 2)) *
            get<kFFT>().hopSize());
  }

  void reset() { mBufferedProcess.reset(); }

private:
  algorithm::NoveltySegmentation mNovelty{get<kMaxKernelSize>(),
                                          get<kMaxFilterSize>()};
  ParameterTrackChanges<index, index, index, double, index, index, double>
                  mParamsTracker;
  BufferedProcess mBufferedProcess;
  algorithm::STFT mSTFT{get<kFFT>().winSize(), get<kFFT>().fftSize(),
                        get<kFFT>().hopSize()};
  FluidTensor<std::complex<double>, 1> mSpectrum;
  FluidTensor<double, 1>               mMagnitude;
  FluidTensor<double, 1>               mBands;
  FluidTensor<double, 1>               mFeature;
  algorithm::MelBands                  mMelBands;
  algorithm::DCT                       mDCT{40, 13};
  algorithm::YINFFT                    mYinFFT;
  algorithm::Loudness                  mLoudness{get<kMaxFFTSize>()};
};

auto constexpr NRTNoveltySliceParams = makeNRTParams<NoveltySliceClient>(
    InputBufferParam("source", "Source Buffer"),
    BufferParam("indices", "Indices Buffer"));
template <typename T>
using NRTNoveltySliceClient =
    NRTSliceAdaptor<NoveltySliceClient<T>, decltype(NRTNoveltySliceParams),
                    NRTNoveltySliceParams, 1, 1>;

template <typename T>
using NRTThreadingNoveltySliceClient =
    NRTThreadingAdaptor<NRTNoveltySliceClient<T>>;

} // namespace client
} // namespace fluid
