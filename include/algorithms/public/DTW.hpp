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

#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidDataSet.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <cstddef>
#include <iterator>
#include <random>

namespace fluid {
namespace algorithm {


enum class DTWConstraint { kUnconstrained, kIkatura, kSakoeChiba };

// debt of gratitude to the wonderful article on
// https://rtavenar.github.io/blog/dtw.html a better explanation of DTW than any
// other algorithm explanation I've seen

class DTW
{
  struct Constraint;

public:
  explicit DTW() = default;
  ~DTW() = default;

  void init(index p = 2) { mPNorm = p; }
  void clear() { mCalculated = false; }

  index           size() const { return mPNorm; }
  constexpr index dims() const { return 0; }
  constexpr index initialized() const { return mInitialized; }

  double process(InputRealMatrixView x1, InputRealMatrixView x2,
                 DTWConstraint c = DTWConstraint::kUnconstrained,
                 Allocator&    alloc = FluidDefaultAllocator())
  {
    ScopedEigenMap<Eigen::VectorXd> x1r(x1.cols(), alloc),
        x2r(x2.cols(), alloc);
    Constraint constraint(c, x1.rows(), x2.rows());

    mDistanceMetrics.resize(x1.rows(), x2.rows());
    mDistanceMetrics.fill(std::numeric_limits<double>::max());

    // simple brute force DTW is very inefficient, see FastDTW
    for (index i = constraint.rowStart(); i < constraint.rowEnd(); i++)
    {
      for (index j = constraint.colStart(i); j < constraint.colEnd(i); j++)
      {
        x1r = _impl::asEigen<Eigen::Matrix>(x1.row(i));
        x2r = _impl::asEigen<Eigen::Matrix>(x2.row(j));

        mDistanceMetrics(i, j) = differencePNormToTheP(x1r, x2r);

        if (i > 0 || j > 0)
        {
          double minimum = std::numeric_limits<double>::max();

          if (i > 0) minimum = std::min(minimum, mDistanceMetrics(i - 1, j));
          if (j > 0) minimum = std::min(minimum, mDistanceMetrics(i, j - 1));
          if (i > 0 && j > 0)
            minimum = std::min(minimum, mDistanceMetrics(i - 1, j - 1));

          mDistanceMetrics(i, j) += minimum;
        }
      }
    }

    mCalculated = true;

    return std::pow(mDistanceMetrics(x1.rows() - 1, x2.rows() - 1),
                    1.0 / mPNorm);
  }

private:
  RealMatrix mDistanceMetrics;
  index      mPNorm{2};

  bool       mCalculated{false};
  const bool mInitialized{true};

  // P-Norm of the difference vector
  // Lp{vec} = (|vec[0]|^p + |vec[1]|^p + ... + |vec[n-1]|^p + |vec[n]|^p)^(1/p)
  // i.e., the 2-norm of a vector is the euclidian distance from the origin
  //       the 1-norm is the sum of the absolute value of the elements
  // To the power P since we'll be summing multiple Norms together and they
  // can combine into a single norm if you calculate the norm of multiple norms
  // (normception)
  inline double
  differencePNormToTheP(const Eigen::Ref<const Eigen::VectorXd>& v1,
                        const Eigen::Ref<const Eigen::VectorXd>& v2)
  {
    // assert(v1.size() == v2.size());
    return (v1.array() - v2.array()).abs().pow(mPNorm).sum();
  }

  struct Constraint
  {
    Constraint(DTWConstraint c, index rows, index cols)
        : mType{c}, mRows{rows}, mCols{cols} {};

    const index rowStart() const { return 0; };
    const index rowEnd() const { return mRows; };

    index colStart(index row)
    {
      switch (mType)
      {
      case DTWConstraint::kUnconstrained: return 0;

      case DTWConstraint::kIkatura: break;

      case DTWConstraint::kSakoeChiba: break;

      default: return -1;
      }
    };

    index colEnd()
    {
      switch (mType)
      {
      case DTWConstraint::kUnconstrained: return mCols;

      case DTWConstraint::kIkatura: break;

      case DTWConstraint::kSakoeChiba: break;

      default: return -1;
      }
    };

  private:
    DTWConstraint mType;
    index         mRows, mCols;

    inline static index rasterLineMinY(float x1, float x2, float y1, float y2,
                                       float x)
    {
      return y1 + (x - x1) * ((y2 - y1) / (x2 - x1));
    }

    inline static index rasterLineMaxY(float x1, float x2, float y1, float y2,
                                       float x)
    {
      if (y2 + x1 > y1 + x2)
        return y1 + (x - x1 + 1) * ((y2 - y1) / (x2 - x1)) - 1;
      else
        return y1 + (x - x1) * ((y2 - y1) / (x2 - x1));
    }
  }; // struct Constraint
};

} // namespace algorithm
} // namespace fluid