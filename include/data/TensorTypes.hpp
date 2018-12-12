#pragma once

#include "FluidTensor.hpp"
#include <complex>

namespace fluid {
  using std::complex;

  using RealMatrix = FluidTensor<double, 2>;
  using ComplexMatrix = FluidTensor<complex<double>, 2>;
  using RealVector = FluidTensor<double, 1>;
  using ComplexVector = FluidTensor<complex<double>, 1>;

  using RealMatrixView = FluidTensorView<double, 2>;
  using ComplexMatrixView = FluidTensorView<complex<double>, 2>;
  using RealVectorView = FluidTensorView<double, 1>;
  using ComplexVectorView = FluidTensorView<complex<double>, 1>;

} // namespace fluid
