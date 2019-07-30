
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
                    double minFreq, double maxFreq, double sampleRate) {
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
    if(maxFreq==0) maxFreq = 1;
    if(minFreq==0) minFreq = 1;
    yin(0) = 1;
    double tmpSum = 0;
    for (int i = 1; i < nBins; i++) {
      tmpSum += yin(i);
      yin(i) *= i / tmpSum;
    }
    double pitch = sampleRate / minFreq;
    double pitchConfidence = 0;
    if (tmpSum > 0) {
      ArrayXd yinFlip = -yin;
      // segment from max to min freq
      int minBin = std::round(sampleRate / maxFreq);
      int maxBin = std::round(sampleRate / minFreq);
      if(minBin > yinFlip.size() - 1) minBin =  yinFlip.size() - 1;
      if(maxBin > yinFlip.size() - minBin - 1) maxBin =  yinFlip.size() - minBin - 1;
      if(maxBin > minBin){
        yinFlip = yinFlip.segment(minBin, maxBin - minBin);
        auto vec = pd.process(yinFlip, 1, yinFlip.minCoeff());
        if (vec.size() > 0) {
          pitch = sampleRate / (minBin + vec[0].first);
          pitchConfidence = 1 + vec[0].second;
        }
      }
    }
    output(0) = pitch;
    output(1) = pitchConfidence;
  }
};
} // namespace algorithm
} // namespace fluid
