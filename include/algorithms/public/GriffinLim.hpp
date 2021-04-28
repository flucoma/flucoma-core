#pragma once

#include "algorithms/util/FluidEigenMappings.hpp"
#include "algorithms/util/AlgorithmUtils.hpp"
#include "algorithms/public/STFT.hpp"
#include "data/TensorTypes.hpp"
#include "data/FluidIndex.hpp"
#include <Eigen/Core>
#include <cmath>

namespace fluid {
namespace algorithm {

class GriffinLim {

public:
  void process(ComplexMatrixView in, index nSamples, index nIter,
               index winSize, index fftSize, index hopSize) {
    using namespace Eigen;
    using namespace _impl;
    using namespace std::complex_literals;
    double momentum = 0.9;
    auto stft = STFT(winSize, fftSize, hopSize);
    auto istft = ISTFT(winSize, fftSize, hopSize);
    ArrayXd tmp = ArrayXd::Zero(nSamples);
    ArrayXXcd magnitude = asEigen<Array>(in).abs();
    ArrayXXcd phase = ArrayXXcd::Random(magnitude.rows(), magnitude.cols()) * 2 * 1i * pi;
    phase = phase.exp();
    ArrayXXcd estimate =  ArrayXXcd::Zero(magnitude.rows(), magnitude.cols());
    ArrayXXcd prev = ArrayXXcd::Zero(magnitude.rows(), magnitude.cols());
    for (index i = 0; i < nIter; i++) {
      prev = estimate;
      ArrayXXcd spectrogram = magnitude * phase;
      istft.process(asFluid(spectrogram), asFluid(tmp));
      stft.process(asFluid(tmp), asFluid(estimate));
      phase = estimate - (momentum / (1 + momentum)) * prev;
      phase = phase / (phase.abs() + epsilon);
    }
    estimate = magnitude * phase;
    in = asFluid(estimate);
  }
};
} // namespace algorithm
} // namespace fluid
