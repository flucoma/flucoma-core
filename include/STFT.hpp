#pragma once

#include <Eigen/Core>
#include <FFT.hpp>
#include <FluidEigenMappings.hpp>
#include <FluidTensor.hpp>
#include <Windows.hpp>
#include <algorithm>
#include <numeric>
#include <vector>

namespace fluid {
namespace stft {

using Eigen::Map;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;

using fft::FFT;
using fft::IFFT;

using windows::windowFuncs;
using windows::WindowType;

using ComplexMatrix = FluidTensor<std::complex<double>, 2>;
using ComplexVector = FluidTensor<std::complex<double>, 1>;
using RealMatrix = FluidTensor<double, 2>;
using RealVector = FluidTensor<double, 1>;

using fluid::eigenmappings::FluidToMatrixXcd;
using fluid::eigenmappings::MatrixXdToFluid;

struct Spectrogram {
  ComplexMatrix mData;
  Spectrogram(ComplexMatrix data) : mData(data) {}
  RealMatrix getMagnitude() const {
    MatrixXcd dataMat = FluidToMatrixXcd(this->mData)();
    MatrixXd magRealMat = dataMat.cwiseAbs().real();
    return MatrixXdToFluid(magRealMat)();
  }

  RealMatrix getPhase() const; // TODO

  long nFrames() const { return this->mData.extent(0); }
  long nBins() const { return this->mData.extent(1); }
};

class STFT {
public:
  STFT(size_t windowSize, size_t fftSize, size_t hopSize)
      : mWindowSize(windowSize), mHopSize(hopSize), mFrameSize(fftSize / 2 + 1),
        mWindow(windowFuncs[WindowType::Hann](windowSize)), mFFT(fftSize) {}

  Spectrogram process(const RealVector audio) {
    int halfWindow = mWindowSize / 2;
    mWorkBuf = std::vector<double>(mWindowSize, 0);
    RealVector padded(audio.size() + mWindowSize + mHopSize);
    padded(slice(halfWindow, audio.size())) = audio;
    int nFrames = floor((padded.size() - mWindowSize) / mHopSize);
    ComplexMatrix data(nFrames, mFrameSize);
    Spectrogram result(data);
    for (int i = 0; i < nFrames; i++) {
      int start = i * mHopSize;
      int end = start + mWindowSize;
      auto spectrum = processFrame(padded(slice(start, end, 1)));
      result.mData.row(i) = spectrum(slice(0, mFrameSize));
    }
    return result;
  }

private:
  size_t mWindowSize;
  size_t mHopSize;
  size_t mFrameSize;
  std::vector<double> mWindow;
  std::vector<double> mWorkBuf;
  FFT mFFT;

  ComplexVector processFrame(const RealVector &frame) {
    for (int i = 0; i < mWindowSize; i++) {
      mWorkBuf[i] = frame[i] * mWindow[i];
    }
    std::vector<std::complex<double>> spectrum = mFFT.process(mWorkBuf);
    return ComplexVector(spectrum);
  }
};

class ISTFT {
public:
  ISTFT(size_t windowSize, size_t fftSize, size_t hopSize)
      : mWindowSize(windowSize), mHopSize(hopSize), mFrameSize(fftSize / 2 + 1),
        mWindow(windowFuncs[WindowType::Hann](windowSize)),
        mScale(1 / double(fftSize)), mIFFT(fftSize) {}

  RealVector process(const Spectrogram &spec) {
    int halfWindow = mWindowSize / 2;
    int outputSize = mWindowSize + (spec.nFrames() - 1) * mHopSize;
    outputSize += mWindowSize + mHopSize;
    std::vector<double> outputPadded(outputSize, 0);
    std::vector<double> norm(outputSize, 0);

    for (int i = 0; i < spec.nFrames(); i++) {
      std::vector<std::complex<double>> tmp(
          spec.mData.row(i).data(), spec.mData.row(i).data() + mFrameSize);
      std::vector<double> frame = mIFFT.process(tmp);
      for (int j = 0; j < mWindowSize; j++) {
        outputPadded[i * mHopSize + j] += frame[j] * mScale * mWindow[j];
        norm[i * mHopSize + j] += mWindow[j] * mWindow[j];
      }
    }
    std::transform(norm.begin(), norm.end(), norm.begin(),
                   [](double x) { return x < 1e-10 ? 1 : x; });
    for (int i = 0; i < outputSize; i++) {
      outputPadded[i] /= norm[i];
    }
    std::vector<double> result(outputPadded.begin() + halfWindow,
                               outputPadded.end() - halfWindow - mHopSize);
    return RealVector(result);
  }

private:
  size_t mWindowSize;
  size_t mHopSize;
  size_t mFrameSize;
  std::vector<double> mWindow;
  double mScale;
  IFFT mIFFT;
};

} // namespace stft
} // namespace fluid
