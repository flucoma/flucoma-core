#pragma once

#include "../../data/TensorTypes.hpp"
#include "../util/FluidEigenMappings.hpp"

#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class DCT {
public:
  using ArrayXd = Eigen::ArrayXd;
  using MatrixXd = Eigen::MatrixXd;

  void init(int inputSize, int outputSize) {
    using std::sqrt;
    assert(inputSize >= outputSize);
    mInputSize = inputSize;
    mOutputSize = outputSize;
    mTable = MatrixXd::Zero(mOutputSize, mInputSize);
    for (int i = 0; i < mOutputSize; i++) {
      double scale = i == 0 ? 1.0 / sqrt(inputSize) : sqrt(2.0 / inputSize);
      ArrayXd freqs = ((M_PI / inputSize) * i) *
                      ArrayXd::LinSpaced(inputSize, 0.5, inputSize - 0.5);
      mTable.row(i) = freqs.cos() * scale;
    }
  }
  void processFrame(const RealVector in, RealVectorView out) {
    assert(in.size() == mInputSize);
    ArrayXd frame = _impl::asEigen<Eigen::Array>(in);
    ArrayXd result = (mTable * frame.matrix()).array();
    out = _impl::asFluid(result);
  }

  void processFrame(Eigen::Ref<const ArrayXd> input, Eigen::Ref<ArrayXd> output) {
    output= (mTable * input.matrix()).array();
  }
  int mInputSize{40};
  int mOutputSize{13};
  MatrixXd mTable;
};
}; // namespace algorithm
}; // namespace fluid
