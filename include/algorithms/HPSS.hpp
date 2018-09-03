#pragma once

#include "MedianFilter.hpp"
#include "data/FluidEigenMappings.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

namespace fluid {
namespace hpss {

using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::Map;
using Eigen::MatrixXd;
using std::vector;

using fluid::eigenmappings::ArrayXXdToFluid;
using fluid::eigenmappings::FluidToArrayXXd;
using fluid::eigenmappings::FluidToMatrixXd;
using fluid::eigenmappings::MatrixXdToFluid;

using RealMatrix = FluidTensor<double, 2>;
using RealVector = FluidTensor<double, 1>;

using fluid::slice;
using fluid::medianfilter::MedianFilter;

struct HPSSModel {
  MatrixXd H;
  MatrixXd P;

  RealMatrix getHarmonicEstimate() const {
    return MatrixXdToFluid(H.transpose())();
  }

  RealMatrix getPercussiveEstimate() const {
    return MatrixXdToFluid(P.transpose())();
  }

  RealMatrix getMixEstimate() const {
    return MatrixXdToFluid((H + P).transpose())();
  }
};

class HPSS {
public:
  HPSS(int vSize, int hSize)
      : mVSize(vSize), mHSize(hSize), mVMedianFilter(vSize),
        mHMedianFilter(hSize) {}


  const HPSSModel process(const RealMatrix &X){
    const auto &epsilon = std::numeric_limits<double>::epsilon;
    HPSSModel result;
    int nFrames = X.extent(0);
    int nBins = X.extent(1);
    int paddedH = mHSize + mHSize * ceil(nFrames / double(mHSize));
    int paddedV = mVSize + mVSize * ceil(nBins / double(mVSize));
    MatrixXd tmp = MatrixXd::Zero(paddedV, paddedH);
    MatrixXd H = MatrixXd::Zero(paddedV, paddedH);
    MatrixXd V = MatrixXd::Zero(paddedV, paddedH);
    ArrayXXd tmp1 = FluidToMatrixXd(X)().transpose();
    tmp.block((mVSize - 1) / 2, (mHSize - 1) / 2, nBins, nFrames) = tmp1;
    for (int i = mHSize / 2; i < nFrames +  mHSize / 2; i++) {
      mVMedianFilter.process(tmp.col(i).array(), V.col(i).array());
    }
    ArrayXd tmpRow(paddedH);
    for (int i = mVSize / 2; i < nBins + mVSize / 2; i++) {
      mHMedianFilter.process(tmp.row(i).transpose().array(), tmpRow);
      H.row(i) = tmpRow.transpose();
    }
    result.H = H.block((mVSize - 1) / 2, (mHSize - 1) / 2, nBins, nFrames);
    result.P = V.block((mVSize - 1) / 2, (mHSize - 1) / 2, nBins, nFrames);
    return result;
  }

private:
  size_t mVSize;
  size_t mHSize;
  MedianFilter mVMedianFilter;
  MedianFilter mHMedianFilter;
};
} // namespace hpss
} // namespace fluid
