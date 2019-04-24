
#pragma once

#include "../../data/TensorTypes.hpp"
#include "../util/FFT.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/PeakDetection.hpp"
#include <Eigen/Core>
#include <fstream>
#include <iostream>

namespace fluid {
namespace algorithm {

using _impl::asEigen;
using _impl::asFluid;
using Eigen::Array;
using Eigen::ArrayXcd;
using Eigen::ArrayXd;

class YINFFT {

public:
  void processFrame(const RealVectorView &input, RealVectorView output,
                    double sampleRate) {
    PeakDetection pd;
    ArrayXd mag = asEigen<Array>(input);
    ArrayXd squareMag = mag.square();
    int nBins = mag.size();
    FFT fft(2 * (mag.size() - 1));
    double squareMagSum = 2 * squareMag.sum();
    ArrayXd squareMagSym(2 * (nBins - 1));
    squareMagSym << squareMag[0], squareMag.segment(1, nBins - 1),
        squareMag.segment(1, nBins - 2).reverse();
    ArrayXcd squareMagFFT = fft.process(squareMagSym);
    ArrayXd yin = squareMagSum - squareMagFFT.real();
    yin(0) = 1;
    double tmpSum = 0;
    for (int i = 1; i < nBins; i++) {
      tmpSum += yin(i);
      yin(i) *= i / tmpSum;
    }
    double pitch = 0;
    double pitchConfidence = 0;
    if (tmpSum > 0) {
      ArrayXd yinFlip = -yin;
      auto vec = pd.process(yinFlip, 1, yinFlip.minCoeff());
      if (vec.size() > 0) {
        pitch = sampleRate / vec[0].first;
        pitchConfidence = 1 + vec[0].second;
      }
    }
    output(0) = pitch;
    output(1) = pitchConfidence;
  }
};
} // namespace algorithm
} // namespace fluid
