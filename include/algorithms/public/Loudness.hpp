/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "../util/AlgorithmUtils.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/KWeightingFilter.hpp"
#include "../util/TruePeak.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <cmath>

namespace fluid {
namespace algorithm {

class Loudness
{

public:
  Loudness(index maxSize) : mTP(maxSize) {}

  void init(index size, double sampleRate)
  {
    mFilter.init(sampleRate);
    mTP.init(size, sampleRate);
    mSize = size;
  }

  void processFrame(const RealVectorView& input, RealVectorView output,
                    bool weighting, bool truePeak)
  {
    using namespace Eigen;
    using namespace std;
    assert(output.size() == 2);
    assert(input.size() == mSize);
    ArrayXd in = _impl::asEigen<Array>(input);
    ArrayXd filtered(mSize);
    for (index i = 0; i < mSize; i++)
      filtered(i) = weighting ? mFilter.processSample(in(i)) : in(i);
    double loudness = -0.691 + 10 * log10(filtered.square().mean() + epsilon);
    double peak = truePeak ? mTP.processFrame(input) : in.abs().maxCoeff();
    peak = 20 * log10(peak + epsilon);
    output(0) = loudness;
    output(1) = peak;
  }

private:
  TruePeak         mTP;
  KWeightingFilter mFilter;
  index            mSize{1024};
};

} // namespace algorithm
} // namespace fluid
