/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See LICENSE file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "../util/AlgorithmUtils.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/KWeightingFilter.hpp"
#include "../util/TruePeak.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Eigen>

namespace fluid {
namespace algorithm {

class Loudness
{

public:
  Loudness(int maxSize) : mTP(maxSize) {}

  void init(int size, int sampleRate)
  {
    mFilter.init(sampleRate);
    mTP.init(size, sampleRate);
    mSize = size;
  }

  void processFrame(const RealVectorView& input, RealVectorView output,
                    bool weighting, bool truePeak)
  {
    using namespace Eigen;
    assert(output.size() == 2);
    assert(input.size() == mSize);
    ArrayXd in = _impl::asEigen<Array>(input);
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
  TruePeak         mTP;
  KWeightingFilter mFilter;
  int              mSize{1024};
};

}; // namespace algorithm
}; // namespace fluid
