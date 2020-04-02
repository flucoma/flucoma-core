#pragma once

#include "algorithms/util/FluidEigenMappings.hpp"
#include "algorithms/public/STFT.hpp"
#include "data/TensorTypes.hpp"
#include <Eigen/Core>
#include <cmath>

namespace fluid {
namespace algorithm {

class GriffinLim {

public:
  void process(ComplexMatrixView in, int nSamples, int nIter,
               int winSize, int fftSize, int hopSize) {
    using namespace Eigen;
    using namespace _impl;
    using namespace std::complex_literals;
    auto stft = STFT(winSize, fftSize, hopSize);
    auto istft = ISTFT(winSize, fftSize, hopSize);
    ArrayXXcd spectrogram = asEigen<Array>(in);
    ArrayXd tmp = ArrayXd::Zero(nSamples);
    ArrayXXcd magnitude = spectrogram.abs();
    ArrayXXcd phase =  ArrayXXcd::Zero(spectrogram.rows(), spectrogram.cols());
    ArrayXXcd estimate =  ArrayXXcd::Zero(spectrogram.rows(), spectrogram.cols());
    for (int i = 0; i < nIter; i++) {
      istft.process(asFluid(spectrogram), asFluid(tmp));
      stft.process(asFluid(tmp), asFluid(phase));
      phase = phase.arg();
      //phase = 1j * phase / (phase.abs() + epsilon);
      spectrogram = magnitude * 1i * phase.exp();
    }
    in = asFluid(spectrogram);
  }
};
} // namespace algorithm
} // namespace fluid
