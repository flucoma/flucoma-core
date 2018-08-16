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

using Eigen::ArrayXd;
using Eigen::ArrayXXcd;
using Eigen::Map;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;

using fft::FFT;
using fft::IFFT;

using windows::windowFuncs;
using windows::WindowType;

using ComplexMatrix = FluidTensor<std::complex<double>, 2>;
using RealMatrix = FluidTensor<double, 2>;
using RealVector = FluidTensor<double, 1>;

using fluid::eigenmappings::FluidToArrayXXcd;
using fluid::eigenmappings::FluidToMatrixXcd;
using fluid::eigenmappings::MatrixXcdToFluid;
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
        mFFT(fftSize) {
    mWindow = Map<ArrayXd>(windowFuncs[WindowType::Hann](mWindowSize).data(),
                           mWindowSize);
  }

  Spectrogram process(const RealVector &audio) {
    int halfWindow = mWindowSize / 2;
    mWorkBuf = std::vector<double>(mWindowSize, 0);
    ArrayXd padded(audio.size() + mWindowSize + mHopSize);
    padded.segment(halfWindow, audio.size()) =
        Map<ArrayXd>(audio.data(), audio.size());
    int nFrames = floor((padded.size() - mWindowSize) / mHopSize);
    MatrixXcd result(nFrames, mFrameSize);
    for (int i = 0; i < nFrames; i++) {
      result.row(i) =
          mFFT.process(padded.segment(i * mHopSize, mWindowSize) * mWindow);
    }
    return ComplexMatrix(MatrixXcdToFluid(result)());
  }

private:
  size_t mWindowSize;
  size_t mHopSize;
  size_t mFrameSize;
  ArrayXd mWindow;
  std::vector<double> mWorkBuf;
  FFT mFFT;
};

class ISTFT {
public:
  ISTFT(size_t windowSize, size_t fftSize, size_t hopSize)
      : mWindowSize(windowSize), mHopSize(hopSize), mFrameSize(fftSize / 2 + 1),
        mScale(1 / double(fftSize)), mIFFT(fftSize) {
    mWindow = Map<ArrayXd>(windowFuncs[WindowType::Hann](mWindowSize).data(),
                           mWindowSize);
  }

  RealVector process(const Spectrogram &spec) {
    const auto &epsilon = std::numeric_limits<double>::epsilon;
    int halfWindow = mWindowSize / 2;
    int outputSize = mWindowSize + (spec.nFrames() - 1) * mHopSize;
    outputSize += mWindowSize + mHopSize;
    ArrayXXcd specData = FluidToArrayXXcd(spec.mData)();
    ArrayXd outputPadded = ArrayXd::Zero(outputSize);
    ArrayXd norm = ArrayXd::Zero(outputSize);
    for (int i = 0; i < spec.nFrames(); i++) {
      ArrayXd frame = mIFFT.process(specData.row(i));
      outputPadded.segment(i * mHopSize, mWindowSize) +=
          frame * mScale * mWindow;
      norm.segment(i * mHopSize, mWindowSize) += mWindow * mWindow;
    }
    outputPadded = outputPadded / norm.max(epsilon());
    return RealVector(
        outputPadded
            .segment(halfWindow, outputPadded.size() - halfWindow - mHopSize)
            .data(),
        outputSize);
  }

private:
  size_t mWindowSize;
  size_t mHopSize;
  size_t mFrameSize;
  ArrayXd mWindow;
  double mScale;
  IFFT mIFFT;
};

} // namespace stft
} // namespace fluid
