
#pragma once

#include "../../data/TensorTypes.hpp"
#include "../util/FluidEigenMappings.hpp"
#include <Eigen/Core>
#include <fstream>
#include <iostream>

namespace fluid {
namespace algorithm {

using _impl::asEigen;
using _impl::asFluid;
using Eigen::Array;
using Eigen::ArrayXd;

class HPS {

public:
  void processFrame(const RealVectorView &input, RealVectorView output,
                    int nHarmonics, double sampleRate) {
    ArrayXd mag = asEigen<Array>(input);
    ArrayXd hps = mag;
    int nBins = mag.size();

    for(int i = 2; i < nHarmonics; i++){
      int hBins = nBins/i;
      ArrayXd h = ArrayXd::Zero(hBins);
      for(int j = 0; j < hBins; j++) h(j) = mag(j * i);
      ArrayXd hp = ArrayXd::Zero(nBins);
      hp.segment(0, hBins) = h;
      hps = hps * hp;
    }
    ArrayXd::Index maxIndex;
    double confidence = hps.sum() ==0? 0 : hps.maxCoeff(&maxIndex) / hps.sum();
    double f0 = maxIndex * sampleRate / ((nBins - 1) * 2);
    output(0) = f0;
    output(1) = confidence;
  }
};
} // namespace algorithm
} // namespace fluid
