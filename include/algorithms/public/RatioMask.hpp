/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
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
#include "../../data/FluidMemory.hpp"
#include <Eigen/Core>

namespace fluid {
namespace algorithm {

class RatioMask
{

  using ArrayXXd = Eigen::ArrayXXd;

public:
  RatioMask(index maxRows, index maxCols, Allocator& alloc)
      : mMultiplier(maxRows, maxCols, alloc)
  {}

  void init(RealMatrixView denominator)
  {
    using namespace _impl;
    using namespace Eigen;
    mRows = denominator.rows();
    mCols = denominator.cols();
    mMultiplier.topLeftCorner(mRows, mCols) =
        (1 / asEigen<Array>(denominator).max(epsilon));
    mInitialized = true;
  }

  void process(const ComplexMatrixView& mixture, RealMatrixView targetMag,
               index exponent, ComplexMatrixView out)
  {
    using namespace _impl;
    using namespace Eigen;
    assert(mInitialized);
    assert(mixture.cols() == targetMag.cols());
    assert(mixture.rows() == targetMag.rows());
    asEigen<Array>(out) =
        asEigen<Array>(mixture) *
        (asEigen<Array>(targetMag).pow(exponent) *
         mMultiplier.topLeftCorner(mRows, mCols).pow(exponent))
            .min(1.0);
  }

private:
  ScopedEigenMap<ArrayXXd> mMultiplier;
  bool                     mInitialized{false};
  index                    mRows;
  index                    mCols;
};

} // namespace algorithm
} // namespace fluid
