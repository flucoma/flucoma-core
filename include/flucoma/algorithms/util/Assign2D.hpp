/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

// Modified Jonker-Volgenant Algorithm
// DF Crouse. On implementing 2D rectangular assignment algorithms.
// IEEE Transactions on Aerospace and Electronic Systems, 52(4):1679-1696, August 2016,

#pragma once

#include "AlgorithmUtils.hpp"
#include "FluidEigenMappings.hpp"
#include "../../data/FluidIndex.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class Assign2D
{
public:
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXXd = Eigen::ArrayXXd;
  using ArrayXi = Eigen::ArrayXi;
  static constexpr index UNASSIGNED = -1;

  bool process(Eigen::Ref<const Eigen::ArrayXXd> costMatrix,
               Eigen::Ref<ArrayXidx>             result)
  {
    using namespace std;
    using namespace Eigen;
    double minCost = costMatrix.minCoeff();
    if(minCost > 0) minCost = 0;
    mCost = costMatrix - minCost;
    if(mCost.cols() > mCost.rows()){
      mCost.transposeInPlace();
      mTransposed = true;
    }
    else mTransposed = false;

    mRow2Col = ArrayXidx::Constant(mCost.rows(), UNASSIGNED);
    mCol2Row = ArrayXidx::Constant(mCost.cols(), UNASSIGNED);
    mV = ArrayXd::Zero(mCost.rows());
    mU = ArrayXd::Zero(mCost.cols());
    for (index c = 0; c < mCost.cols(); c++){
      index sink = shortestPath(c);
      if(sink == UNASSIGNED) return false;
      index j = sink;
      index h;
      while(true){
        index i = mPred(j);
        mRow2Col(j) = i;
        h = mCol2Row(i);
        mCol2Row(i) = j;
        j = h;
        if(i == c) break;
      }
    }
    result = mTransposed?mCol2Row:mRow2Col;
    return true;
  }

  index shortestPath(index c){
    using namespace std;
    mPred = ArrayXidx::Zero(mCost.rows());
    ArrayXidx scannedCols = ArrayXidx::Zero(mCost.cols());
    ArrayXidx scannedRows = ArrayXidx::Zero(mCost.rows());
    vector<index> row2Scan(asUnsigned(mCost.rows()));
    iota(row2Scan.begin(), row2Scan.end(), 0);
    index nRows2Scan = mCost.rows();
    index sink = UNASSIGNED;
    double delta = 0;
    index currentCol = c;
    index currentRow, currentClosestRow, closestRow;
    double minVal, reducedCost;
    ArrayXd shortestPathCost = ArrayXd::Constant(mCost.rows(), infinity);
    while(sink == UNASSIGNED){
      scannedCols(currentCol) = 1;
      minVal = infinity;
      for(index r = 0; r < nRows2Scan; r++){
        currentRow = row2Scan[asUnsigned(r)];
        reducedCost = delta
                    + mCost(currentRow, currentCol)
                    - mU(currentCol) - mV(currentRow);
        if(reducedCost < shortestPathCost(currentRow)){
          mPred(currentRow) = currentCol;
          shortestPathCost(currentRow) = reducedCost;
        }
        if(shortestPathCost(currentRow) < minVal){
          minVal = shortestPathCost(currentRow);
          currentClosestRow = r;
        }
      }
      if(isinf(minVal)) return UNASSIGNED;
      closestRow = row2Scan[asUnsigned(currentClosestRow)];
      scannedRows(closestRow) = 1;
      nRows2Scan--;
      row2Scan.erase(row2Scan.begin() + currentClosestRow);
      delta = shortestPathCost(closestRow);
      if(mRow2Col(closestRow) == UNASSIGNED) sink = closestRow;
      else currentCol = mRow2Col(closestRow);
    }
    mU(c) = mU(c) + delta;
    for(index i = 0; i < mCost.cols(); i++){
      if(i !=c && scannedCols(i) != 0){
        mU(i) = mU(i) + delta - shortestPathCost(mCol2Row(i));
      }
    }

    for(index i = 0; i < mCost.rows(); i++){
      if(scannedRows(i) != 0){
        mV(i) = mV(i) - delta + shortestPathCost(i);
      }
    }
    return sink;
}
private:
  ArrayXXd mCost;
  ArrayXidx  mRow2Col;
  ArrayXidx  mCol2Row;
  ArrayXd  mU;
  ArrayXd  mV;
  ArrayXidx mPred;
  bool mTransposed{false};
};

constexpr index Assign2D::UNASSIGNED;

} // namespace algorithm
} // namespace fluid
