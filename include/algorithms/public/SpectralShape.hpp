
#pragma once

#include "../../data/TensorTypes.hpp"
#include "../util/Descriptors.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "Windows.hpp"
#include <Eigen/Eigen>
#include <fstream>
#include <iostream>

namespace fluid {
namespace algorithm {

using _impl::asEigen;
using _impl::asFluid;
using Eigen::ArrayXXd;
using Eigen::Array;
using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::Ref;
using Eigen::RowVectorXd;
using Eigen::VectorXd;

class SpectralShape {
  double const epsilon = std::numeric_limits<double>::epsilon();

public:
  SpectralShape(size_t maxFrame) : mMagBuffer(maxFrame), mOutputBuffer(7) {}

  /*void process(const RealMatrix &input, RealVector output) {
    int nFrames = input.rows();
    int nFeatures = 7;
    ArrayXXd X = asEigen<Array>(input);
    ArrayXXd O = ArrayXXd::Zero(nFrames, nFeatures);
    clock_t begin = std::clock();
    for (int i = 0; i < nFrames; i++) {
      ArrayXd in = X.row(i) + epsilon;
      ArrayXd out = ArrayXd::Zero(nFeatures);
      processFrame(in);
      O.row(i) = out;
    }
    // std::cout<<O<<std::endl;
    // clock_t end = std::clock();
    // double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    RowVectorXd mean = O.colwise().mean();
    RowVectorXd std = (O.rowwise() - mean.array()).colwise().mean();
    ArrayXXd diff = O.block(1, 0, nFrames - 1, nFeatures) -
                    O.block(0, 0, nFrames - 1, nFeatures);
    RowVectorXd dmean = diff.colwise().mean();
    RowVectorXd dstd = (diff.rowwise() - mean.array()).colwise().mean();
    ArrayXd result = ArrayXd::Zero(nFeatures * 4);
    result.segment(0, 7) = mean;
    result.segment(7, 7) = dmean;
    result.segment(14, 7) = std;
    result.segment(21, 7) = dstd;
    output = asFluid(result);
  }*/

  void processFrame(Ref<ArrayXd> x) {
    //assert(output.size() == 7); // TODO
    int size = x.size();
    double xSum = x.sum();
    ArrayXd lin = ArrayXd::LinSpaced(size, 0, size - 1);
    double centroid = (x * lin).sum() / xSum;
    double spread = (x * (lin - centroid).square()).sum() / xSum;
    double skewness =
        (x * (lin - centroid).pow(3)).sum() / (spread * sqrt(spread) * xSum);
    double kurtosis =
        (x * (lin - centroid).pow(4)).sum() / (spread * spread * xSum);
    double flatness = exp(x.log().mean()) / x.mean();
    double rolloff = size - 1;
    double cumSum = 0;
    double target = 0.95 * xSum;
    for (int i = 0; cumSum <= target && i < size; i++) {
      cumSum += x(i);
      if (cumSum > target) {
        rolloff = i - (cumSum - target) / x(i);
        break;
      }
    }
    double crest = x.maxCoeff() / sqrt(x.square().mean());
    mOutputBuffer(0) = centroid;
    mOutputBuffer(1) = spread;
    mOutputBuffer(2) = skewness;
    mOutputBuffer(3) = kurtosis;
    mOutputBuffer(4) = flatness;
    mOutputBuffer(5) = rolloff;
    mOutputBuffer(6) = crest;
  }

  void processFrame(const RealVector &input, RealVectorView output) {
    assert(output.size() == 7); // TODO
    ArrayXd in = asEigen<Array>(input);
    processFrame(in);
    output = asFluid(mOutputBuffer);
  }

private:
  ArrayXd mMagBuffer;
  ArrayXd mOutputBuffer;
};

}; // namespace algorithm
}; // namespace fluid
