/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
// http://csclab.murraystate.edu/~bob.pilgrim/445/munkres.html

#pragma once

#include "AlgorithmUtils.hpp"
#include "FluidEigenMappings.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class Munkres
{
public:
  using intPair = std::pair<int, int>;

  Munkres(index rows, index cols, Allocator& alloc)
      : mN{std::max(rows, cols)}, mCost{mN, mN, alloc}, mRowMin{mN, alloc},
        mColMin{mN, alloc}, mMask{mN, mN, alloc}, mPath{2 * mN + 1, 2, alloc},
        mRowCover{mN, alloc}, mColCover{mN, alloc}
  {
    //    reset();
  }

  //  void init(index rows, index cols)
  //  {
  //    using namespace Eigen;
  //    index N = std::max(rows, cols);
  //    mCost = ArrayXXd::Zero(N, N);
  //    mRowMin = ArrayXd::Zero(N);
  //    mColMin = ArrayXd::Zero(N);
  //    mMask = ArrayXXi::Zero(N, N);
  //    mRowCover = ArrayXi::Zero(N);
  //    mColCover = ArrayXi::Zero(N);
  //    mPath = ArrayXXi::Zero(2 * N + 1, 2);
  //  }

  void reset()
  {
    mCost.setZero();
    mRowMin.setZero();
    mColMin.setZero();
    mMask.setZero();
    mRowCover.setZero();
    mColCover.setZero();
    mPath.setZero();
  }

  void process(Eigen::Ref<const Eigen::ArrayXXd> costMatrix,
               Eigen::Ref<Eigen::ArrayXi> result, Allocator& alloc)
  {
    bool done;
    reset();
    double maxCost = costMatrix.maxCoeff();
    mCost = Eigen::ArrayXXd::Ones(mCost.rows(), mCost.cols());
    mCost = mCost * (10.0 * maxCost);
    mCost.block(0, 0, costMatrix.rows(), costMatrix.cols()) = costMatrix;
    step1();
    step2();
    done = step3();
    while (!done)
    {
      intPair Z = step4(alloc);
      while (Z.first < 0)
      {
        step6();
        Z = step4(alloc);
      }
      step5(Z, alloc);
      done = step3();
    }
    for (int i = 0; i < result.size(); i++)
    {
      mMask.row(i).maxCoeff(&result(i));
    }
  }

  void step1()
  {
    mRowMin = mCost.rowwise().minCoeff();
    mColMin = (mCost.colwise() - mRowMin).colwise().minCoeff();
    for (int i = 0; i < mCost.rows(); i++)
    {
      double min = mCost.row(i).minCoeff();
      mCost.row(i) -= min;
    }
  }

  void step2()
  {
    for (int i = 0; i < mCost.rows(); i++)
    {
      for (int j = 0; j < mCost.cols(); j++)
      {
        if (mCost(i, j) == 0 && mRowCover(i) == 0 && mColCover(j) == 0)
        {
          mMask(i, j) = 1;
          mRowCover(i) = 1;
          mColCover(j) = 1;
        }
      }
    }
    mRowCover.setZero();
    mColCover.setZero();
  }

  bool step3()
  {
    for (int i = 0; i < mMask.rows(); i++)
    {
      for (int j = 0; j < mMask.cols(); j++)
      {
        if (mMask(i, j) == 1) { mColCover(j) = 1; }
      }
    }
    Eigen::Index nCovered = (mColCover == 1).count();
    return nCovered >= mMask.rows() || nCovered >= mMask.cols();
  }

  intPair step4(Allocator& alloc)
  {
    int                            row = -1, col = -1;
    intPair                        result = std::make_pair(row, col);
    ScopedEigenMap<Eigen::ArrayXi> r(mMask.cols(), alloc);
    while (true)
    {
      intPair pos = findZero();
      row = pos.first;
      col = pos.second;
      if (row < 0) { break; }
      mMask(row, col) = 2;
      r = mMask.row(row);
      int colStar = findValue(r, 1);
      if (colStar >= 0)
      {
        col = colStar;
        mRowCover(row) = 1;
        mColCover(col) = 0;
      }
      else
      {
        result = std::make_pair(row, col);
        break;
      }
    }
    return result;
  }


  void step5(const intPair Z, Allocator& alloc)
  {
    int row = -1, col = -1;
    int pathCount = 0;
    mPath(pathCount, 0) = Z.first;
    mPath(pathCount, 1) = Z.second;
    ScopedEigenMap<Eigen::ArrayXi> tmpCol(mMask.rows(), alloc);
    ScopedEigenMap<Eigen::ArrayXi> r(mMask.cols(), alloc);
    while (true)
    {
      int tmp = mPath(pathCount, 1);
      tmpCol = mMask.col(tmp);
      row = findValue(tmpCol, 1);
      if (row == -1) break;
      pathCount++;
      mPath(pathCount, 0) = row;
      mPath(pathCount, 1) = mPath(pathCount - 1, 1);
      r = mMask.row(mPath(pathCount, 0));
      col = findValue(r, 2);
      pathCount++;
      mPath(pathCount, 0) = mPath(pathCount - 1, 0);
      mPath(pathCount, 1) = col;
    }
    augmentPath(pathCount);
    mRowCover.setZero();
    mColCover.setZero();
    erasePrimes();
  }

  void step6()
  {
    double m = minCost();
    for (int i = 0; i < mCost.rows(); i++)
    {
      for (int j = 0; j < mCost.cols(); j++)
      {
        if (mRowCover(i) == 1) mCost(i, j) += m;
        if (mColCover(j) == 0) mCost(i, j) -= m;
      }
    }
  }

  double minCost()
  {
    double minVal = std::numeric_limits<double>::max();
    for (int i = 0; i < mCost.rows(); i++)
    {
      for (int j = 0; j < mCost.cols(); j++)
      {
        if (mRowCover(i) == 0 && mColCover(j) == 0)
        {
          double v = mCost(i, j);
          minVal = (minVal > v) ? v : minVal;
        }
      }
    }
    return minVal;
  }

  void erasePrimes()
  {
    for (int i = 0; i < mMask.rows(); i++)
    {
      for (int j = 0; j < mMask.cols(); j++)
      {
        if (mMask(i, j) == 2) { mMask(i, j) = 0; }
      }
    }
  }

  intPair findZero()
  {
    for (int i = 0; i < mCost.rows(); i++)
      for (int j = 0; j < mCost.cols(); j++)
      {
        if (mCost(i, j) == 0 && mRowCover(i) == 0 && mColCover(j) == 0)
          return std::make_pair(i, j);
      }
    return std::make_pair(-1, -1);
  }

  int findValue(Eigen::Ref<const Eigen::ArrayXi> vector, const int val)
  {
    for (int i = 0; i < vector.size(); i++)
    {
      if (vector(i) == val) return i;
    }
    return -1;
  }

  void augmentPath(const int count)
  {
    for (int i = 0; i < count + 1; i++)
    {
      int c = mPath(i, 0), r = mPath(i, 1);
      mMask(c, r) = mMask(c, r) == 1 ? 0 : 1;
    }
  }

private:
  index                           mN;
  ScopedEigenMap<Eigen::ArrayXXd> mCost;
  ScopedEigenMap<Eigen::ArrayXd>  mRowMin;
  ScopedEigenMap<Eigen::ArrayXd>  mColMin;
  ScopedEigenMap<Eigen::ArrayXXi> mMask;
  ScopedEigenMap<Eigen::ArrayXXi> mPath;
  ScopedEigenMap<Eigen::ArrayXi>  mRowCover;
  ScopedEigenMap<Eigen::ArrayXi>  mColCover;
};
} // namespace algorithm
} // namespace fluid
