/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
// http://csclab.murraystate.edu/~bob.pilgrim/445/munkres.html
#pragma once

#include "../util/AlgorithmUtils.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidIndex.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class Munkres
{
public:
  using intPair = std::pair<int, int>;
  void init(index rows, index cols)
  {
    using namespace Eigen;
    index N = std::max(rows, cols);
    mCost = ArrayXXd::Zero(N, N);
    mRowMin = ArrayXd::Zero(N);
    mColMin = ArrayXd::Zero(N);
    mMask = ArrayXXi::Zero(N, N);
    mRowCover = ArrayXi::Zero(N);
    mColCover = ArrayXi::Zero(N);
    mPath = ArrayXXi::Zero(2 * N + 1, 2);
  }

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
               Eigen::Ref<Eigen::ArrayXi>        result)
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
      intPair Z = step4();
      while (Z.first < 0)
      {
        step6();
        Z = step4();
      }
      step5(Z);
      done = step3();
    }
    for (int i = 0; i < result.size(); i++)
    { mMask.row(i).maxCoeff(&result(i)); }
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

  intPair step4()
  {
    int     row = -1, col = -1;
    intPair result = std::make_pair(row, col);
    while (true)
    {
      intPair pos = findZero();
      row = pos.first;
      col = pos.second;
      if (row < 0) { break; }
      mMask(row, col) = 2;
      Eigen::ArrayXi r = mMask.row(row);
      int            colStar = findValue(r, 1);
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


  void step5(const intPair Z)
  {
    int row = -1, col = -1;
    int pathCount = 0;
    mPath(pathCount, 0) = Z.first;
    mPath(pathCount, 1) = Z.second;
    while (true)
    {
      int            tmp = mPath(pathCount, 1);
      Eigen::ArrayXi tmpCol = mMask.col(tmp);
      row = findValue(tmpCol, 1);
      if (row == -1) break;
      pathCount++;
      mPath(pathCount, 0) = row;
      mPath(pathCount, 1) = mPath(pathCount - 1, 1);
      Eigen::ArrayXi r = mMask.row(mPath(pathCount, 0));
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
  Eigen::ArrayXXd mCost;
  Eigen::ArrayXd  mRowMin;
  Eigen::ArrayXd  mColMin;
  Eigen::ArrayXXi mMask;
  Eigen::ArrayXXi mPath;
  Eigen::ArrayXi  mRowCover;
  Eigen::ArrayXi  mColCover;
};
} // namespace algorithm
} // namespace fluid
