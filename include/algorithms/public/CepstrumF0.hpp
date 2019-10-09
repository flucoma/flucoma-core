
#pragma once

#include "../util/DCT.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/PeakDetection.hpp"
#include "../util/AlgorithmUtils.hpp"
#include "../../data/TensorTypes.hpp"

#include <Eigen/Eigen>

namespace fluid {
namespace algorithm {

class CepstrumF0 {

public:
  using ArrayXd = Eigen::ArrayXd;

  void init(int size){
    mNumBins = size;
    mDCT.init(mNumBins, mNumBins);
    mCepstrum = ArrayXd(mNumBins);
  }

  void processFrame(const RealVectorView &input, RealVectorView output,
                    double minFreq, double maxFreq, double sampleRate) {
    using namespace Eigen;
    PeakDetection pd;

    ArrayXd mag = _impl::asEigen<Array>(input);
    ArrayXd logMag = mag.max(epsilon).log();
    mDCT.processFrame(logMag, mCepstrum);
    double pitch = sampleRate / minFreq;
    double confidence = 0;

    int minBin = std::round(sampleRate / maxFreq);
    int maxBin = std::round(sampleRate / minFreq);
    if(maxBin > minBin && maxBin < mCepstrum.size()){
      auto vec = pd.process(mCepstrum.segment(minBin, maxBin - minBin ), 1);
      if(vec.size() > 0) {
        pitch = sampleRate / (vec[0].first + minBin);
        confidence = vec[0].second / mCepstrum[0];
      }
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
