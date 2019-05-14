
#pragma once

#include "../../data/TensorTypes.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/KWeightingFilter.hpp"
#include "../util/TruePeak.hpp"
#include <Eigen/Eigen>
#include <fstream>
#include <iostream>

namespace fluid {
namespace algorithm {

using _impl::asEigen;
using _impl::asFluid;
using Eigen::Array;
using Eigen::ArrayXd;

using algorithm::KWeightingFilter;
using algorithm::TruePeak;

class Loudness {

public:
  Loudness(int maxSize) : mTP(maxSize) {}

  void init(int size, int sampleRate) {
    mFilter.init(sampleRate);
    mTP.init(size, sampleRate);
    mSize = size;
  }

  void processFrame(const RealVectorView &input, RealVectorView output,
                    bool weighting, bool truePeak) {
    assert(output.size() == 2);
    assert(input.size() == mSize);
    double const epsilon = std::numeric_limits<double>::epsilon();
    ArrayXd in = asEigen<Array>(input);
    ArrayXd filtered(mSize);
    for (int i = 0; i < mSize; i++)
      filtered(i) = weighting ? mFilter.processSample(in(i)) : in(i);
    double loudness =
        -0.691 + 10 * std::log10(filtered.square().mean() + epsilon);
    double peak = truePeak ? mTP.processFrame(input) : in.abs().maxCoeff();
    peak = 20 * std::log10(peak + epsilon);
    output(0) = loudness;
    output(1) = peak;
  }

private:
  TruePeak mTP;
  KWeightingFilter mFilter;
  int mSize;
};

}; // namespace algorithm
}; // namespace fluid
