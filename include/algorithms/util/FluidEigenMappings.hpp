#pragma once

#include "../../data/FluidTensor.hpp"
#include <Eigen/Core>

namespace fluid {
namespace algorithm {

using Eigen::Array;
using Eigen::Dynamic;
using Eigen::Map;
using Eigen::Matrix;
using Eigen::RowMajor;
using Eigen::StorageOptions;
using std::complex;

template <typename T, template <typename, int, int, int, int, int> class E,
          int R = Dynamic, int C = Dynamic, StorageOptions S = RowMajor,
          int M = Dynamic>
class EigenMatrixMap {
public:
  using MapType = Map<E<T, R, C, S, M, M>>;
  using ConstMapType = const MapType;
  EigenMatrixMap(const FluidTensorView<T, 2> &x) : mMatrix(x) {}
  ConstMapType operator()() const {
    return ConstMapType(mMatrix.data(), mMatrix.extent(0), mMatrix.extent(1));
  }

  MapType operator()() {
    return MapType(mMatrix.data(), mMatrix.extent(0), mMatrix.extent(1));
  }

private:
  FluidTensorView<T, 2> mMatrix;
};

template <typename T, template <typename, int, int, int, int, int> class E,
          int R = Dynamic, int C = Dynamic, StorageOptions S = RowMajor,
          int M = Dynamic>
class FluidMatrixMap {
  using MatrixType = E<T, R, C, S, M, M>;

public:
  FluidMatrixMap(const MatrixType &x) : mMatrix(x) {}
  FluidTensorView<T, 2> operator()() const {
    FluidTensor<T, 2> mapped(mMatrix.rows(), mMatrix.cols());
    Map<MatrixType>(mapped.data(), mMatrix.rows(), mMatrix.cols()) = mMatrix;
    return mapped;
  }

private:
  MatrixType mMatrix;
};

using FluidToMatrixXd = EigenMatrixMap<double, Matrix>;
using FluidToMatrixXcd = EigenMatrixMap<complex<double>, Matrix>;
using MatrixXdToFluid = FluidMatrixMap<double, Matrix>;
using MatrixXcdToFluid = FluidMatrixMap<complex<double>, Matrix>;

using FluidToArrayXXd = EigenMatrixMap<double, Array>;
using FluidToArrayXXcd = EigenMatrixMap<complex<double>, Array>;
using ArrayXXdToFluid = FluidMatrixMap<double, Array>;
using ArrayXXcdToFluid = FluidMatrixMap<complex<double>, Array>;

using ArrayXXdMap = Map<Array<double, Dynamic, Dynamic, RowMajor>>;
using ArrayXdMap = Map<Array<double, Dynamic, RowMajor>>;
using ArrayXXcdMap = Map<Array<complex<double>, Dynamic, Dynamic, RowMajor>>;
using ArrayXXcdConstMap =
    Map<const Array<complex<double>, Dynamic, Dynamic, RowMajor>>;
using ArrayXXdConstMap = Map<const Array<double, Dynamic, Dynamic, RowMajor>>;
using ArrayXcdMap = Map<Array<complex<double>, Dynamic, RowMajor>>;
using ArrayXdConstMap = Map<const Array<double, Dynamic, RowMajor>>;
using ArrayXcdConstMap = Map<const Array<complex<double>, Dynamic, RowMajor>>;

} // namespace algorithm
} // namespace fluid
