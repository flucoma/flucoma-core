
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
  void processFrame(const RealVectorView &input, RealVectorView output,
                    double sampleRate) {
    DCT dct;
    PeakDetection pd;
    int nBins = input.size();
    const auto &epsilon = std::numeric_limits<double>::epsilon();
    ArrayXd mag = asEigen<Array>(input);
    ArrayXd logMag = mag.max(epsilon).log();
    dct.init(nBins, nBins);
    ArrayXd cepstrum = ArrayXd(nBins);
    dct.processFrame(logMag, cepstrum);
    auto vec = pd.process(cepstrum, 2);
    double pitch = sampleRate / vec[1].first;
    double confidence = vec[1].second /cepstrum[0];
    output(0) = pitch;
    output(1) = confidence;
  }
};
} // namespace algorithm
} // namespace fluid
