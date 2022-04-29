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
#include "../../data/FluidIndex.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>

namespace fluid {
namespace algorithm {

class RatioMask
{

  using ArrayXXd = Eigen::ArrayXXd;

public:
  void init(RealMatrixView denominator)
  {
    using namespace _impl;
    using namespace Eigen;
    mMultiplier = (1 / asEigen<Array>(denominator).max(epsilon));
  }

  void process(const ComplexMatrixView& mixture, RealMatrixView targetMag,
               index exponent, ComplexMatrixView result)
  {
    using namespace _impl;
    using namespace Eigen;
    assert(mixture.cols() == targetMag.cols());
    assert(mixture.rows() == targetMag.rows());
    ArrayXXcd tmp =
        asEigen<Array>(mixture) *
        (asEigen<Array>(targetMag).pow(exponent) * mMultiplier.pow(exponent))
            .min(1.0);
    result <<= asFluid(tmp);
  }

private:
  ArrayXXd mMultiplier;
};

} // namespace algorithm
} // namespace fluid
