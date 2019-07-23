#pragma once

#include "BufferedProcess.hpp"
#include "../common/AudioClient.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../nrt/FluidNRTClientWrapper.hpp"
#include "../../algorithms/public/Loudness.hpp"
#include "../../algorithms/public/RTNoveltySegmentation.hpp"
#include "../../algorithms/public/STFT.hpp"
#include "../../algorithms/public/YINFFT.hpp"
#include "../../algorithms/util/DCT.hpp"
#include "../../algorithms/util/MelBands.hpp"
#include "../../algorithms/util/TruePeak.hpp"
#include "../../data/TensorTypes.hpp"

#include <tuple>

namespace fluid {
namespace client {

using algorithm::DCT;
using algorithm::Loudness;
using algorithm::MelBands;
using algorithm::STFT;
using algorithm::TruePeak;
using algorithm::YINFFT;

using algorithm::RTNoveltySegmentation;

enum NoveltyParamIndex {
  kFeature,
  kKernelSize,
  kThreshold,
  kFilterSize,
  kFFT,
  kMaxFFTSize,
  kMaxKernelSize,
  kMaxFilterSize,
};

auto constexpr NoveltyParams = defineParameters(
    EnumParam("feature", "Feature", 0, "Spectrum", "MFCC", "Pitch", "Loudness"),
    LongParam("kernelSize", "KernelSize", 3, Min(3), Odd(),
              UpperLimit<kMaxKernelSize>()),
    FloatParam("threshold", "Threshold", 0.5, Min(0)),
    LongParam("filterSize", "Smoothing Filter Size", 1, Min(1),
              UpperLimit<kMaxFilterSize>()),
    FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4),
                           PowerOfTwo{}),
    LongParam<Fixed<true>>("maxKernelSize", "Maxiumm Kernel Size", 101, Min(3),
                           Odd()),
    LongParam<Fixed<true>>("maxFilterSize", "Maxiumm Filter Size", 100,
                           Min(1)));

template <typename T>
class NoveltySlice
    : public FluidBaseClient<decltype(NoveltyParams), NoveltyParams>,
      public AudioIn,
      public AudioOut {

  using HostVector = HostVector<T>;

public:
  NoveltySlice(ParamSetViewType &p) : FluidBaseClient(p) {
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::audioChannelsOut(1);
  }

  void process(std::vector<HostVector> &input, std::vector<HostVector> &output,
               bool reset = false) {
    using algorithm::RTNoveltySegmentation;
    using std::size_t;

    if (!input[0].data() || !output[0].data())
      return;

    size_t hostVecSize = input[0].size();
    size_t windowSize = get<kFFT>().winSize();
    int feature = get<kFeature>();
    if (mParamsTracker.changed(hostVecSize, get<kFeature>(), get<kKernelSize>(),
                               get<kThreshold>(), get<kFilterSize>(),
                               windowSize)) {
      mBufferedProcess.hostSize(hostVecSize);
      mBufferedProcess.maxSize(windowSize, windowSize,
                               FluidBaseClient::audioChannelsIn(),
                               FluidBaseClient::audioChannelsOut());
      int nDims = 2;
      if (feature < 3) {
        mSpectrum.resize(get<kFFT>().frameSize());
        mMagnitude.resize(get<kFFT>().frameSize());
        mSTFT = STFT(get<kFFT>().winSize(), get<kFFT>().fftSize(),
                     get<kFFT>().hopSize());
      }
      if (feature == 0) {
        nDims = get<kFFT>().frameSize();
      } else if (feature == 1) {
        mBands.resize(40);
        mMelBands.init(20, 2000, 40, get<kFFT>().frameSize(), sampleRate(),
                       true);
        mDCT.init(40, 13);
        nDims = 13;
      } else if (feature == 3) {
        mLoudness.init(windowSize, sampleRate());
      }
      mFeature.resize(nDims);
      mNovelty.init(get<kKernelSize>(), get<kThreshold>(), get<kFilterSize>(),
                    nDims);
    }
    RealMatrix in(1, hostVecSize);
    in.row(0) = input[0];
    RealMatrix out(1, hostVecSize);
    int frameOffset = 0; // in case kHopSize < hostVecSize
    mBufferedProcess.push(RealMatrixView(in));
    mBufferedProcess.process(
        windowSize, windowSize, get<kFFT>().hopSize(), reset,
        [&, this](RealMatrixView in, RealMatrixView) {
          switch (feature) {
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
          out.row(0)(frameOffset) = mNovelty.processFrame(mFeature);
          frameOffset += get<kFFT>().hopSize();
        });
    output[0] = out.row(0);
  }

  long latency() {
    return get<kFFT>().winSize() +
           ((get<kKernelSize>() - 1) / 2) * get<kFFT>().hopSize() +
           (get<kFilterSize>() - 1) * get<kFFT>().hopSize();
  }

private:
  RTNoveltySegmentation mNovelty{get<kMaxKernelSize>(), get<kMaxFilterSize>()};
  ParameterTrackChanges<size_t, size_t, size_t, double, size_t, size_t>
      mParamsTracker;
  BufferedProcess mBufferedProcess;
  STFT mSTFT{get<kFFT>().winSize(), get<kFFT>().fftSize(),
             get<kFFT>().hopSize()};
  MelBands mMelBands;
  DCT mDCT;
  FluidTensor<std::complex<double>, 1> mSpectrum;
  FluidTensor<double, 1> mMagnitude;
  FluidTensor<double, 1> mBands;
  FluidTensor<double, 1> mFeature;
  YINFFT mYinFFT;
  Loudness mLoudness{get<kMaxFFTSize>()};
};

auto constexpr NRTNoveltySliceParams =
    makeNRTParams<NoveltySlice>({InputBufferParam("source", "Source Buffer")},
                                  {BufferParam("indices", "Indices Buffer")});
template <typename T>
using NRTNoveltySlice =
    NRTSliceAdaptor<NoveltySlice<T>, decltype(NRTNoveltySliceParams),
                    NRTNoveltySliceParams, 1, 1>;

} // namespace client
} // namespace fluid
