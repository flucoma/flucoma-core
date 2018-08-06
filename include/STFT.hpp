#pragma once

#include <Eigen/Core>
#include <FFT.hpp>
#include <FluidTensor.hpp>
#include <Windows.hpp>
#include <algorithm>
#include <numeric>
#include <vector>

namespace fluid {
namespace stft {

using std::complex;
using std::rotate;
using std::transform;
using std::vector;

using Eigen::Map;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;

using fft::FFT;
using fft::IFFT;

using windows::windowFuncs;
using windows::WindowType;

//using FluidTensor;
//using slice;

using ComplexMatrix = FluidTensor<complex<double>, 2>;
using ComplexVector = FluidTensor<complex<double>, 1>;
using RealMatrix = FluidTensor<double, 2>;
using RealVector = FluidTensor<double, 1>;
using MatrixXcdMap = Map<Eigen::Matrix<complex<double>, Eigen::Dynamic,
                                       Eigen::Dynamic, Eigen::RowMajor>>;
using MatrixXdMap =
    Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

struct Spectrogram {
  ComplexMatrix mData;
  Spectrogram(ComplexMatrix data) : mData(data) {}

  RealMatrix getMagnitude() const {
    MatrixXcdMap dataMat(this->mData.data(), this->mData.extent(0),
                         this->mData.extent(1));
    MatrixXd magRealMat = dataMat.cwiseAbs().real();
    RealMatrix result(this->mData.extent(0), this->mData.extent(1));
    MatrixXdMap(result.data(), this->mData.extent(0), this->mData.extent(1)) =
        magRealMat;
    return result;
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
  vector<double> mWindow;
  FFT mFFT;

  ComplexVector processFrame(const RealVector &frame) {
    vector<double> tmp(frame.size(), 0);
    for (int i = 0; i < mWindowSize; i++) {
      tmp[i] = frame(i) * mWindow[i];
    }
    vector<complex<double>> spectrum = mFFT.process(tmp);
    return ComplexVector(spectrum);
  }
};

class ISTFT {
public:
  ISTFT(size_t windowSize, size_t fftSize, size_t hopSize)
      : mWindowSize(windowSize), mHopSize(hopSize), mFrameSize(fftSize / 2 + 1),
        mWindow(windowFuncs[WindowType::Hann](windowSize)),
        mScale(0.5 / double(fftSize)), mIFFT(fftSize) {}

  RealVector process(const Spectrogram &spec) {
    int halfWindow = mWindowSize / 2;
    int outputSize = mWindowSize + (spec.nFrames() - 1) * mHopSize;
    outputSize += mWindowSize + mHopSize;
    vector<double> outputPadded(outputSize, 0);
    vector<double> norm(outputSize, 0);

    for (int i = 0; i < spec.nFrames(); i++) {
      vector<complex<double>> tmp(spec.mData.row(i).data(),
                                  spec.mData.row(i).data() + mFrameSize);
      vector<double> frame = mIFFT.process(tmp);
      for (int j = 0; j < mWindowSize; j++) {
        outputPadded[i * mHopSize + j] += frame[j] * mScale * mWindow[j];
        norm[i * mHopSize + j] += mWindow[j] * mWindow[j];
      }
    }
    transform(norm.begin(), norm.end(), norm.begin(),
              [](double x) { return x < 1e-10 ? 1 : x; });
    for (int i = 0; i < outputSize; i++) {
      outputPadded[i] /= norm[i];
    }
    vector<double> result(outputPadded.begin() + halfWindow,
                          outputPadded.end() - halfWindow - mHopSize);
    return RealVector(result);
  }

private:
  size_t mWindowSize;
  size_t mHopSize;
  size_t mFrameSize;
  vector<double> mWindow;
  double mScale;
  IFFT mIFFT;
};

} // namespace stft
} // namespace fluid
