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

  void init() {}
  void clear() {}

  index           size() const { return mPNorm; }
  constexpr index dims() const { return 0; }
  constexpr index initialized() const { return true; }

  double process(InputRealMatrixView x1, InputRealMatrixView x2,
                 DTWConstraint constr = DTWConstraint::kUnconstrained,
                 index         param = 2,
                 Allocator&    alloc = FluidDefaultAllocator()) const
  {
    ScopedEigenMap<Eigen::VectorXd> x1r(x1.cols(), alloc),
        x2r(x2.cols(), alloc);
    Constraint constraint(constr, x1.rows(), x2.rows(), param);

    mDistanceMetrics.resize(x1.rows(), x2.rows());
    mDistanceMetrics.fill(std::numeric_limits<double>::max());

    constraint.iterate([&, this](index r, index c) {
      x1r = _impl::asEigen<Eigen::Matrix>(x1.row(r));
      x2r = _impl::asEigen<Eigen::Matrix>(x2.row(c));

      mDistanceMetrics(r, c) = differencePNormToTheP(x1r, x2r);

      if (r > 0 || c > 0)
      {
        double minimum = std::numeric_limits<double>::max();

        if (r > 0) minimum = std::min(minimum, mDistanceMetrics(r - 1, c));
        if (c > 0) minimum = std::min(minimum, mDistanceMetrics(r, c - 1));
        if (r > 0 && c > 0)
          minimum = std::min(minimum, mDistanceMetrics(r - 1, c - 1));

        mDistanceMetrics(r, c) += minimum;
      }
    });

    return std::pow(mDistanceMetrics(x1.rows() - 1, x2.rows() - 1),
                    1.0 / mPNorm);
  }

private:
  mutable RealMatrix mDistanceMetrics;
  const index        mPNorm{2};

  // P-Norm of the difference vector
  // Lp{vec} = (|vec[0]|^p + |vec[1]|^p + ... + |vec[n-1]|^p + |vec[n]|^p)^(1/p)
  // i.e., the 2-norm of a vector is the euclidian distance from the origin
  //       the 1-norm is the sum of the absolute value of the elements
  // To the power P since we'll be summing multiple Norms together and they
  // can combine into a single norm if you calculate the norm of multiple norms
  // (normception)
  inline double
  differencePNormToTheP(const Eigen::Ref<const Eigen::VectorXd>& v1,
                        const Eigen::Ref<const Eigen::VectorXd>& v2) const
  {
    // assert(v1.size() == v2.size());
    return (v1.array() - v2.array()).abs().pow(mPNorm).sum();
  }

  // fun little fold operation to do a variadic minimum
  template <typename... Args>
  inline static auto min(Args&&... args)
  {
    auto m = (args, ...);
    return ((m = std::min(m, args)), ...);
  }

  // filter for minimum chaining, if cond evaluates to false then the value
  // isn't used (never will be the minimum if its the numeric maximum)
  template <typename T>
  inline static T useIf(bool cond, T val)
  {
    return cond ? val : std::numeric_limits<T>::max();
  }

  struct Constraint
  {
    Constraint(DTWConstraint c, index rows, index cols, float param)
        : mType{c}, mRows{rows}, mCols{cols}, mParam{param}
    {
      // ifn't gradient more than digonal set it to be the diagonal
      // (sakoe-chiba with radius 0)
      if (c == DTWConstraint::kIkatura)
      {
        float big = std::max(mRows, mCols), smol = std::min(mRows, mCols);

        if (mParam <= big / smol)
        {
          mType = DTWConstraint::kSakoeChiba;
          mParam = 0;
        }
      }
    };

    void iterate(std::function<void(index, index)> f)
    {
      index first, last;

      for (index r = 0; r < mRows; ++r)
      {
        first = firstCol(r);
        last = lastCol(r);

        for (index c = first; c <= last; ++c) f(r, c);
      }
    };

  private:
    DTWConstraint mType;
    index         mRows, mCols;
    float         mParam; // mParam is either radius (SC) or gradient (Ik)

    inline static index rasterLineMinY(index x1, index y1, float dydx, index x)
    {
      return std::round(y1 + (x - x1) * dydx);
    }

    inline static index rasterLineMinY(index x1, index y1, index x2, index y2,
                                       index x)
    {
      float dy = y2 - y1, dx = x2 - x1;
      return rasterLineMinY(x1, y1, dy / dx, x);
    }

    inline static index rasterLineMaxY(index x1, index y1, float dydx, index x)
    {
      if (dydx > 1)
        return rasterLineMinY(x1, y1, dydx, x + 1) - 1;
      else
        return rasterLineMinY(x1, y1, dydx, x);
    }

    inline static index rasterLineMaxY(index x1, index y1, index x2, index y2,
                                       index x)
    {
      float dy = y2 - y1, dx = x2 - x1;
      return rasterLineMaxY(x1, y1, dy / dx, x);
    }

    index firstCol(index row)
    {
      switch (mType)
      {
      case DTWConstraint::kUnconstrained: return 0;

      case DTWConstraint::kIkatura: {
        index colNorm = rasterLineMinY(mRows - 1, mCols - 1, mParam, row);
        index colInv = rasterLineMinY(0, 0, 1 / mParam, row);

        index col = std::max(colNorm, colInv);

        return col < 0 ? 0 : col > mCols - 1 ? mCols - 1 : col;
      }

      case DTWConstraint::kSakoeChiba: {
        index col = rasterLineMinY(mParam, -mParam, mRows - 1 + mParam,
                                   mCols - 1 - mParam, row);

        return col < 0 ? 0 : col > mCols - 1 ? mCols - 1 : col;
      }
      }
    };

    index lastCol(index row)
    {
      switch (mType)
      {
      case DTWConstraint::kUnconstrained: return mCols - 1;

      case DTWConstraint::kIkatura: {
        index colNorm = rasterLineMaxY(0, 0, mParam, row);
        index colInv = rasterLineMaxY(mRows - 1, mCols - 1, 1 / mParam, row);

        index col = std::min(colNorm, colInv);

        return col < 0 ? 0 : col > mCols - 1 ? mCols - 1 : col;
      }

      case DTWConstraint::kSakoeChiba: {
        index col = rasterLineMaxY(-mParam, mParam, mRows - 1 - mParam,
                                   mCols - 1 + mParam, row);

        return col < 0 ? 0 : col > mCols - 1 ? mCols - 1 : col;
      }
      }
    };
  }; // struct Constraint
};

} // namespace algorithm
} // namespace fluid