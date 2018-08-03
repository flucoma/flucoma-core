#pragma once

#include "FluidTensor.hpp"
#include "Windows.hpp"
#include <Eigen/Core>
#include <HISSTools_FFT/HISSTools_FFT.h>
#include <iostream>
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

using fluid::windows::windowFuncs;
using fluid::windows::WindowType;

using fluid::FluidTensor;
using fluid::slice;

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
  STFT(int windowSize, int fftSize, int hopSize)
      : mWindowSize(windowSize), mFFTSize(fftSize), mHopSize(hopSize),
        mFrameSize(fftSize / 2), mLog2Size(log2(mFFTSize)),
        mWindow(windowFuncs[WindowType::Hann](windowSize)) {
    int log2Size = (int)log2(mFFTSize);
    hisstools_create_setup(&mSetup, log2Size);
    mSplit.realp = new double[(1 << (log2Size - 1)) + 1];
    mSplit.imagp = new double[(1 << (log2Size - 1)) + 1];
  }

  ~STFT() {
    if (mSplit.realp)
      delete mSplit.realp;
    if (mSplit.imagp)
      delete mSplit.imagp;
  }

  Spectrogram process(RealVector audio) {
    int halfWindow = mWindowSize / 2;
    RealVector padded(audio.size() + mWindowSize + mHopSize);
    padded(slice(halfWindow, audio.size())) = audio(slice(0, audio.size()));
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
  int mWindowSize;
  int mFFTSize;
  int mHopSize;
  int mFrameSize;
  int mLog2Size;
  vector<double> mWindow;
  FFT_SETUP_D mSetup;
  FFT_SPLIT_COMPLEX_D mSplit;

  ComplexVector processFrame(const RealVector &frame) {
    vector<double> tmp(frame.size(), 0);
    vector<complex<double>> spectrum(frame.size(), 0);
    for (int i = 0; i < mWindowSize; i++) {
      tmp[i] = frame(i) * mWindow[i];
    }
    hisstools_rfft(mSetup, tmp.data(), &mSplit, mFFTSize, mLog2Size);
    mSplit.realp[mFrameSize] = mSplit.imagp[0];
    mSplit.imagp[mFrameSize] = 0;
    mSplit.imagp[0] = 0;
    for (int i = 0; i < mFrameSize; i++) {
      spectrum[i] = complex<double>(mSplit.realp[i], mSplit.imagp[i]);
    }
    return ComplexVector(spectrum);
  }
};

class ISTFT {
public:
  ISTFT(int windowSize, int fftSize, int hopSize)
      : mWindowSize(windowSize), mFFTSize(fftSize), mHopSize(hopSize),
        mFrameSize(fftSize / 2), mLog2Size(log2(mFFTSize)),
        mScale(0.5 / double(fftSize)),
        mWindow(windowFuncs[WindowType::Hann](windowSize)) {
    hisstools_create_setup(&mSetup, mLog2Size);
    mSplit.realp = new double[(1 << (mLog2Size - 1)) + 1];
    mSplit.imagp = new double[(1 << (mLog2Size - 1)) + 1];
  }

  ~ISTFT() {
    if (mSplit.realp)
      delete mSplit.realp;
    if (mSplit.imagp)
      delete mSplit.imagp;
  }

  RealVector process(Spectrogram spec) {
    int halfWindow = mWindowSize / 2;
    int outputSize = mWindowSize + (spec.nFrames() - 1) * mHopSize;
    outputSize += mWindowSize + mHopSize;
    vector<double> outputPadded(outputSize, 0);
    vector<double> norm(outputSize, 0);

    for (int i = 0; i < spec.nFrames(); i++) {
      RealVector frame = processFrame(spec.mData.row(i));
      for (int j = 0; j < mWindowSize; j++) {
        outputPadded[i * mHopSize + j] += frame(j) * mScale * mWindow[j];
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
  int mWindowSize;
  int mFFTSize;
  int mHopSize;
  int mFrameSize;
  int mLog2Size;
  double mScale;

  vector<double> mWindow;
  FFT_SETUP_D mSetup;
  FFT_SPLIT_COMPLEX_D mSplit;
  const RealVector processFrame(const ComplexVector frame) {
    RealVector result(mFFTSize);

    for (int i = 0; i < frame.size(); i++) {
      mSplit.realp[i] = frame(i).real();
      mSplit.imagp[i] = frame(i).imag();
    }
    mSplit.imagp[0] = mSplit.realp[mFrameSize];
    hisstools_rifft(mSetup, &mSplit, result.data(), mLog2Size);
    return result;
  }
};
} // namespace stft
} // namespace fluid
