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

#include "../util/FluidEigenMappings.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class RunningStats
{
public:
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXXd = Eigen::ArrayXXd;

  void init(index historySize, index inputSize)
  {
    mInputBuffer = ArrayXXd::Zero(inputSize, historySize + 1);
    mInputSquaredBuffer = ArrayXXd::Zero(inputSize, historySize + 1);
    mXSum = ArrayXd::Zero(inputSize);
    mXSqSum = ArrayXd::Zero(inputSize);
    mX1 = ArrayXd::Zero(inputSize);
    mY1 = ArrayXd::Zero(inputSize);
    mInputSquared = ArrayXd::Zero(inputSize);
    mCleanedInput = ArrayXd::Zero(inputSize);
    mHead = 0;
    mN = 0;
    mSize = historySize;
  }

  void process(RealVectorView in, RealVectorView meanOut,
               RealVectorView stdDevOut)
  {
    // Moving average and _sample_ standard deviation
    // https://www.dsprelated.com/showthread/comp.dsp/97276-1.php

    // oldest inputs and squared inputs from buffer
    mX1 = mInputBuffer.col(mHead);
    mY1 = mInputSquaredBuffer.col(mHead);
    
    using MapXd = decltype(_impl::asEigen<Eigen::Array>(in));
    
    MapXd inMap = _impl::asEigen<Eigen::Array>(in);
    mCleanedInput = inMap.isNaN().select(0,inMap);
    mInputSquared = mCleanedInput.square();

    // running sums
    mXSum += (mCleanedInput - mX1);
    mXSqSum += (mInputSquared - mY1);

    // calculate stats
    mN = std::min(mN + 1, mSize);
    _impl::asEigen<Eigen::Array>(meanOut) = mXSum / mN;
    if(mN > 1)
    {
      _impl::asEigen<Eigen::Array>(stdDevOut) = ((mN * mXSqSum - mXSum.square()) / (mN * (mN - 1))).sqrt();
    }
    else
    {
      std::fill(stdDevOut.begin(),stdDevOut.end(),0);
    }

    // write new data
    mInputBuffer.col(mHead) = mCleanedInput;
    mInputSquaredBuffer.col(mHead) = mInputSquared;

    // move on
    mHead++;
    if (mHead >= mInputBuffer.cols() - 1) mHead = 0;
  }

private:
  ArrayXXd mInputBuffer;
  ArrayXXd mInputSquaredBuffer;
  ArrayXd  mXSum;
  ArrayXd  mXSqSum;
  ArrayXd  mX1;
  ArrayXd  mY1;
  ArrayXd  mCleanedInput;
  ArrayXd  mInputSquared;
  index    mHead;
  index    mN;
  index    mSize;
};
} // namespace algorithm
} // namespace fluid
