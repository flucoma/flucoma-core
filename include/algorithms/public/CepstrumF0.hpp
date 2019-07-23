
#pragma once

#include "../../data/TensorTypes.hpp"
#include "../util/DCT.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/PeakDetection.hpp"
#include <Eigen/Eigen>
#include <fstream>
#include <iostream>

namespace fluid {
namespace algorithm {

using _impl::asEigen;
using _impl::asFluid;
using Eigen::Array;
using Eigen::ArrayXd;

class CepstrumF0 {

public:
  void init(int size){
    mNumBins = size;
    mDCT.init(mNumBins, mNumBins);
    mCepstrum = ArrayXd(mNumBins);
  }

  void processFrame(const RealVectorView &input, RealVectorView output,
                    double minFreq, double maxFreq, double sampleRate) {

    PeakDetection pd;
    const auto &epsilon = std::numeric_limits<double>::epsilon();
    ArrayXd mag = asEigen<Array>(input);
    ArrayXd logMag = mag.max(epsilon).log();
    mDCT.processFrame(logMag, mCepstrum);
    int minBin = std::round(sampleRate / maxFreq);
    int maxBin = std::round(sampleRate / minFreq);
    auto vec = pd.process(mCepstrum.segment(minBin, maxBin - minBin ), 1);
    double pitch = sampleRate / minBin;
    double confidence = 0;
    if(vec.size() > 0) {
      pitch = sampleRate / (vec[0].first + minBin);
      confidence = vec[0].second / mCepstrum[0];
    }
    output(0) = pitch;
    output(1) = std::min(std::abs(confidence), 1.0);
  }

private:
  DCT mDCT;
  int mNumBins;
  ArrayXd mCepstrum;
};
} // namespace algorithm
} // namespace fluid
