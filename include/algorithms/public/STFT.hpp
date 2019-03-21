#pragma once

#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include "../util/FFT.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "Windows.hpp"
#include <Eigen/Core>

namespace fluid {
namespace algorithm {

using _impl::asEigen;
using _impl::asFluid;
using Eigen::ArrayXd;
using Eigen::ArrayXXcd;
using Eigen::ArrayXcd;
using Eigen::ArrayXXd;
using Eigen::MatrixXcd;
using Eigen::Map;
using Eigen::Array;

class STFT {

public:
  STFT(size_t windowSize, size_t fftSize, size_t hopSize)
      : mWindowSize(windowSize), mHopSize(hopSize), mFrameSize(fftSize / 2 + 1),
        mFFT(fftSize) {
    mWindow = Map<ArrayXd>(windowFuncs[WindowType::kHann](mWindowSize).data(),
                           mWindowSize);
  }

  static void magnitude(const FluidTensorView<complex<double>, 2> &in,
                        FluidTensorView<double, 2> out) {
    ArrayXXd mag = asEigen<Array>(in).abs().real();
    out = asFluid(mag);
  }

  static void magnitude(const FluidTensorView<complex<double>, 1> &in,
                        FluidTensorView<double, 1> out) {
    ArrayXd mag = asEigen<Array>(in).abs().real();
    out = asFluid(mag);
  }


  void process(const RealVectorView audio, ComplexMatrixView spectrogram) {
    int halfWindow = mWindowSize / 2;
    ArrayXd padded(audio.size() + mWindowSize + mHopSize);
    padded.fill(0);
    padded.segment(halfWindow, audio.size()) =
        Map<const ArrayXd>(audio.data(), audio.size());
    int nFrames = floor((padded.size() - mWindowSize) / mHopSize);
    ArrayXXcd result(nFrames, mFrameSize);
    for (int i = 0; i < nFrames; i++) {
      result.row(i) =
          mFFT.process(padded.segment(i * mHopSize, mWindowSize) * mWindow);
    }
    spectrogram = asFluid(result);
  }

  void processFrame(const RealVectorView frame, ComplexVectorView out) {
    assert(frame.size() == mWindowSize);
    ArrayXcd spectrum = mFFT.process(asEigen<Array>(frame) * mWindow);
    out = asFluid(spectrum);
  }

  RealVectorView window() { return RealVectorView(mWindow.data(), 0, mWindowSize); }

private:
  size_t mWindowSize;
  size_t mHopSize;
  size_t mFrameSize;
  ArrayXd mWindow;
  FFT mFFT;
};

class ISTFT {

public:
  ISTFT(size_t windowSize, size_t fftSize, size_t hopSize)
      : mWindowSize(windowSize), mHopSize(hopSize), mFrameSize(fftSize / 2 + 1),
        mScale(1 / double(fftSize)), mIFFT(fftSize), mBuffer(mWindowSize) {
    mWindow = Map<ArrayXd>(windowFuncs[WindowType::kHann](mWindowSize).data(),
                           mWindowSize);
    mWindowSquared = mWindow * mWindow;
  }

  void process(const ComplexMatrixView spectrogram, RealVectorView audio) {
    const auto &epsilon = std::numeric_limits<double>::epsilon;
    int halfWindow = mWindowSize / 2;
    int nFrames = spectrogram.rows();
    int outputSize = mWindowSize + (nFrames - 1) * mHopSize;
    outputSize += mWindowSize + mHopSize;
    ArrayXXcd specData = asEigen<Array>(spectrogram);
    ArrayXd outputPadded = ArrayXd::Zero(outputSize);
    ArrayXd norm = ArrayXd::Zero(outputSize);
    for (int i = 0; i < nFrames; i++) {
      ArrayXd frame = mIFFT.process(specData.row(i)).segment(0, mWindowSize);
      outputPadded.segment(i * mHopSize, mWindowSize) +=
          frame * mScale * mWindow;
      norm.segment(i * mHopSize, mWindowSize) += mWindow * mWindow;
    }
    outputPadded = outputPadded / norm.max(epsilon());
    ArrayXd trimmed = outputPadded.segment(
        halfWindow, audio.size());
    audio = asFluid(trimmed);
  }

  void processFrame(const ComplexVectorView frame, RealVectorView audio) {
    assert(frame.size() == mFrameSize);
    mBuffer = mIFFT.process(asEigen<Array>(frame)).segment(0, mWindowSize) *
              mWindow * mScale;
    audio = asFluid(mBuffer);
  }

  RealVectorView window() { return RealVectorView(mWindow.data(), 0, mWindowSize); }

private:
  size_t mWindowSize;
  size_t mHopSize;
  size_t mFrameSize;
  ArrayXd mWindow;
  ArrayXd mWindowSquared;
  double mScale;
  IFFT mIFFT;
  ArrayXd mBuffer;
};

} // namespace algorithm
} // namespace fluid
