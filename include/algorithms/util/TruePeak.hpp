
#pragma once

#include "../../data/TensorTypes.hpp"
#include "../util/FFT.hpp"
#include "../util/FluidEigenMappings.hpp"
#include <Eigen/Eigen>
#include <cmath>
#include <fstream>
#include <iostream>

namespace fluid {
namespace algorithm {

using _impl::asEigen;
using _impl::asFluid;
using Eigen::Array;
using Eigen::ArrayXcd;
using Eigen::ArrayXd;

class TruePeak {

public:
  TruePeak(int maxSize) : mFFT(maxSize), mIFFT(maxSize * 4) {}

  void init(int size, int sampleRate) {
    mSampleRate = sampleRate;
    mFFTSize = std::pow(2, std::ceil(std::log(size) / std::log(2)));
    mFactor = sampleRate < 96000 ? 4 : 2;
    mFFT.resize(mFFTSize);
    mIFFT.resize(mFFTSize * mFactor);
    mBuffer = ArrayXcd::Zero((mFFTSize * mFactor / 2) + 1);
  }

  double processFrame(const RealVectorView &input) {
    ArrayXd in = asEigen<Array>(input);
    if (mSampleRate >= 192000) {
      return in.abs().maxCoeff();
    } else {
      double peak;
      ArrayXcd transform = mFFT.process(in);
      mBuffer.setZero();
      mBuffer.segment(0, transform.size()) = transform;
      ArrayXd result = mIFFT.process(mBuffer);
      ArrayXd scaled = result.segment(1, input.size() * mFactor) / mFFTSize;
      peak = scaled.abs().maxCoeff();
      return peak;
    }
  }

private:
  FFT mFFT;
  IFFT mIFFT;
  ArrayXcd mBuffer;
  int mSampleRate;
  int mFactor;
  int mFFTSize;
};
}; // namespace algorithm
}; // namespace fluid
