#pragma once

#include "../../data/TensorTypes.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/MedianFilter.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

namespace fluid {
namespace algorithm {

using _impl::asEigen;
using _impl::asFluid;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::MatrixXd;
using std::vector;

class HPSS {
public:
  HPSS(int vSize, int hSize)
      : mVSize(vSize), mHSize(hSize), mVMedianFilter(vSize),
        mHMedianFilter(hSize) {}

  void process(const RealMatrix &in, RealMatrix harm, RealMatrix perc,
               RealMatrix mixEstimate) {
    int nFrames = in.extent(0);
    int nBins = in.extent(1);
    int paddedH = mHSize + mHSize * ceil(nFrames / double(mHSize));
    int paddedV = mVSize + mVSize * ceil(nBins / double(mVSize));
    MatrixXd tmp = MatrixXd::Zero(paddedV, paddedH);
    MatrixXd H = MatrixXd::Zero(paddedV, paddedH);
    MatrixXd P = MatrixXd::Zero(paddedV, paddedH);
    ArrayXXd tmp1 = asEigen<Array>(in).transpose();
    tmp.block((mVSize - 1) / 2, (mHSize - 1) / 2, nBins, nFrames) = tmp1;
    for (int i = mHSize / 2; i < nFrames + mHSize / 2; i++) {
      mVMedianFilter.process(tmp.col(i).array(), P.col(i).array());
    }
    ArrayXd tmpRow(paddedH);
    for (int i = mVSize / 2; i < nBins + mVSize / 2; i++) {
      mHMedianFilter.process(tmp.row(i).transpose().array(), tmpRow);
      H.row(i) = tmpRow.transpose();
    }
    H = H.block((mVSize - 1) / 2, (mHSize - 1) / 2, nBins, nFrames);
    P = P.block((mVSize - 1) / 2, (mHSize - 1) / 2, nBins, nFrames);
    MatrixXd HT = H.transpose();
    MatrixXd PT = P.transpose();
    MatrixXd MT = HT + PT;
    harm = asFluid(HT);
    perc = asFluid(PT);
    mixEstimate = asFluid(MT);
  }

private:
  size_t mVSize;
  size_t mHSize;
  MedianFilter mVMedianFilter;
  MedianFilter mHMedianFilter;
};
} // namespace algorithm
} // namespace fluid
