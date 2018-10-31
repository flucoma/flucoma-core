#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <numeric>
#include <vector>

#include "../../data/FluidEigenMappings.hpp"
#include "../../data/FluidTensor.hpp"
#include "../util/FFT.hpp"
#include "Windows.hpp"

namespace fluid {
namespace algorithm {

using Eigen::Array;
using Eigen::ArrayXd;
using Eigen::ArrayXXcd;
using Eigen::Dynamic;
using Eigen::Map;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::RowMajor;

// using algorithm::FFT;
// using algorithm::IFFT;
//
// using algorithm::windowFuncs;
// using algorithm::WindowType;
using std::complex;

// using fluid::algorithm::FluidToArrayXXcd;
// using fluid::algorithm::FluidToMatrixXcd;
// using fluid::algorithm::MatrixXcdToFluid;
// using fluid::algorithm::MatrixXdToFluid;

struct Spectrogram {
  using ComplexMatrix = FluidTensor<complex<double>, 2>;
  using RealMatrix = FluidTensor<double, 2>;
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
  using ComplexMatrix = FluidTensor<complex<double>, 2>;
  using RealMatrix = FluidTensor<double, 2>;
  using RealVector = FluidTensor<double, 1>;
  using RealVectorView = FluidTensorView<double, 1>;
  using ComplexVectorView = FluidTensorView<complex<double>, 1>;
  using ArrayXXdConstMap = Map<const Array<double, Dynamic, Dynamic, RowMajor>>;

  STFT(size_t windowSize, size_t fftSize, size_t hopSize)
      : mWindowSize(windowSize), mHopSize(hopSize), mFrameSize(fftSize / 2 + 1),
        mFFT(fftSize) {
    mWindow = Map<ArrayXd>(windowFuncs[WindowType::kHann](mWindowSize).data(),
                           mWindowSize);
  }

  Spectrogram process(const RealVector &audio) {
    int halfWindow = mWindowSize / 2;
    ArrayXd padded(audio.size() + mWindowSize + mHopSize);
    padded.fill(0);
    padded.segment(halfWindow, audio.size()) =
        Map<const ArrayXd>(audio.data(), audio.size());
    int nFrames = floor((padded.size() - mWindowSize) / mHopSize);
    MatrixXcd result(nFrames, mFrameSize);
    for (int i = 0; i < nFrames; i++) {
      result.row(i) =
          mFFT.process(padded.segment(i * mHopSize, mWindowSize) * mWindow);
    }
    return ComplexMatrix(MatrixXcdToFluid(result)());
  }

  ComplexVectorView processFrame(const RealVectorView &frame) {
    assert(frame.size() == mWindowSize);
    return ComplexVectorView(
        mFFT.process((ArrayXXdConstMap(frame.data(), mWindowSize, 1) * mWindow))
            .data(),
        0, mFrameSize);
  }

  RealVector window() {
    FluidTensor<double, 1> win(mWindowSize);
    win = FluidTensorView<double, 1>(mWindow.data(), 0, mWindowSize);
    return win;
  }

private:
  size_t mWindowSize;
  size_t mHopSize;
  size_t mFrameSize;
  ArrayXd mWindow;
  FFT mFFT;
};

class ISTFT {

public:
  using ComplexMatrix = FluidTensor<complex<double>, 2>;
  using RealMatrix = FluidTensor<double, 2>;
  using RealVector = FluidTensor<double, 1>;
  using RealVectorView = FluidTensorView<double, 1>;
  using ComplexVectorView = FluidTensorView<complex<double>, 1>;

  using ArrayXXdMap = Map<Array<double, Dynamic, Dynamic, RowMajor>>;
  using ArrayXcdConstMap =
      Map<const Array<complex<double>, Dynamic, Dynamic, RowMajor>>;

  ISTFT(size_t windowSize, size_t fftSize, size_t hopSize)
      : mWindowSize(windowSize), mHopSize(hopSize), mFrameSize(fftSize / 2 + 1),
        mScale(1 / double(fftSize)), mIFFT(fftSize), mBuffer(mWindowSize) {
    mWindow = Map<ArrayXd>(windowFuncs[WindowType::kHann](mWindowSize).data(),
                           mWindowSize);
    mWindowSquared = mWindow * mWindow;
    // The 2nd row of our output will be constant, and contain the squared
    // window, for the normalisation buffer
    //    ArrayXXdMap(mBuffer.row(1).data(), mWindowSize, 1) = mWindowSquared;
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
      ArrayXd frame = mIFFT.process(specData.row(i)).segment(0, mWindowSize);
      outputPadded.segment(i * mHopSize, mWindowSize) +=
          frame * mScale * mWindow;
      norm.segment(i * mHopSize, mWindowSize) += mWindow * mWindow;
    }
    outputPadded = outputPadded / norm.max(epsilon());
    return RealVector(
        outputPadded
            .segment(halfWindow, outputPadded.size() - halfWindow - mHopSize)
            .data(),
        outputSize - halfWindow - mHopSize);
  }

  RealVectorView processFrame(const ComplexVectorView &frame) {
    assert(frame.size() == mFrameSize);
    ArrayXXdMap(mBuffer.data(), mWindowSize, 1) =
        mIFFT.process(ArrayXcdConstMap(frame.data(), mFrameSize, 1))
            .segment(0, mWindowSize) *
        mWindow * mScale;
    return mBuffer;
  }

  RealVector window() {
    FluidTensor<double, 1> win(mWindowSize);
    win = FluidTensorView<double, 1>(mWindow.data(), 0, mWindowSize);
    return win;
  }

private:
  size_t mWindowSize;
  size_t mHopSize;
  size_t mFrameSize;
  ArrayXd mWindow;
  ArrayXd mWindowSquared;
  double mScale;
  IFFT mIFFT;
  RealVector mBuffer;
};

} // namespace algorithm
} // namespace fluid
