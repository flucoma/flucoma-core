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
#include <random>

namespace fluid {
namespace algorithm {


// debt of gratitude to the wonderful article on
// https://rtavenar.github.io/blog/dtw.html a better explanation of DTW than any
// other algorithm explanation I've seen

class DTW
{
public:
  explicit DTW() = default;
  ~DTW() = default;

  void init() const {}
  void clear() { mCalculated = false; }

  constexpr index size() const { return 0; }
  constexpr index dims() const { return 0; }
  constexpr index initialized() const { return true; }

  double process(InputRealMatrixView x1, InputRealMatrixView x2, index p,
                 Allocator& alloc = FluidDefaultAllocator())
  {
    mCalculated = true;

    return calculateDistanceMetrics(x1, x2, p, alloc);
    ;
  }


  double calculateDistanceMetrics(InputRealMatrixView x1,
                                  InputRealMatrixView x2, index p,
                                  Allocator& alloc)
  {
    ScopedEigenMap<Eigen::VectorXd> x1r(x1.cols(), alloc),
        x2r(x2.cols(), alloc);

    mDistanceMetrics.resize(x1.rows(), x2.rows());

    // bool getPath(InputRealMatrixView x1, 
    //              InputRealMatrixView x2,
    //              RealMatrixView path,
    //              index p = 2,
    //              Allocator& alloc = FluidDefaultAllocator())
    // {
    //     calculateDistanceMetrics(x1, x2, p, alloc);
    //     return getPath(path);
    // }

    // bool getPath(RealMatrixView to)
    // {
    //     if (!mCalculated) return false;

    //     index n = 0;
    //     index maxPathLength = mDistanceMetrics.rows() 
    //                         + mDistanceMetrics.cols() - 1;
    //     RealMatrix path(maxPathLength, 2);

    //     for (index i = mDistanceMetrics.rows(), j = mDistanceMetrics.cols(); 
    //                i > 0 || j > 0; ++n)
    //     {
    //         path.row(n) <<= {(double) i, (double) j};

    //         // if at one end of the matrix just go along that edge
    //         if (i == 0 && (--j, true)) continue;
    //         if (j == 0 && (--i, true)) continue;

    //         double upVal =   mDistanceMetrics(i-1, j  );
    //         double leftVal = mDistanceMetrics(i  , j-1);
    //         double diagVal = mDistanceMetrics(i-1, j-1);

    //         double minVal = std::min(std::min(upVal, leftVal), diagVal);

    //         if (minVal == upVal) --i;
    //         else if (minVal == leftVal) --j; 
    //         else if (minVal == upVal) (--i, --j);
    //         else return false;
    //     }

    //     path.row(n++) <<= {0, 0};

    //     to <<= path;
    //     return true;
    // }
    // simple brute force DTW is very inefficient, see FastDTW
    for (index i = 0; i < x1.rows(); i++)
    {
      for (index j = 0; j < x2.rows(); j++)
      {
        x1r = _impl::asEigen<Eigen::Matrix>(x1.row(i));
        x2r = _impl::asEigen<Eigen::Matrix>(x2.row(j));

        mDistanceMetrics(i, j) = differencePNormToTheP(x1r, x2r, p);

        if (i > 0 || j > 0)
        {
          double minimum = std::numeric_limits<double>::max();

          if (i > 0 && j > 0)
            minimum = std::min(minimum, mDistanceMetrics(i - 1, j - 1));
          if (i > 0) minimum = std::min(minimum, mDistanceMetrics(i - 1, j));
          if (j > 0) minimum = std::min(minimum, mDistanceMetrics(i, j - 1));

          mDistanceMetrics(i, j) += minimum;
        }
      }
    }

    return std::pow(mDistanceMetrics(x1.rows() - 1, x2.rows() - 1), 1.0 / p);
  }

private:
  RealMatrix mDistanceMetrics;
  bool       mCalculated{false};

  // P-Norm of the difference vector
  // Lp{vec} = (|vec[0]|^p + |vec[1]|^p + ... + |vec[n-1]|^p + |vec[n]|^p)^(1/p)
  // i.e., the 2-norm of a vector is the euclidian distance from the origin
  //       the 1-norm is the sum of the absolute value of the elements
  // To the power P since we'll be summing multiple Norms together and they
  // can combine into a single norm if you calculate the norm of multiple norms
  // (normception)
  inline static double
  differencePNormToTheP(const Eigen::Ref<const Eigen::VectorXd>& v1,
                        const Eigen::Ref<const Eigen::VectorXd>& v2, index p)
  {
    // assert(v1.size() == v2.size());
    return (v1.array() - v2.array()).abs().pow(p).sum();
  }
};

} // namespace algorithm
} // namespace fluid