#pragma once

#include "FluidTensor.hpp"
#include <complex>

namespace fluid {
  using std::complex;

  using RealMatrix = FluidTensorView<double, 2>;
  using ComplexMatrix = FluidTensorView<complex<double>, 2>;
  using RealVector = FluidTensorView<double, 1>;
  using ComplexVector = FluidTensorView<complex<double>, 1>;
  using ComplexMatrix = FluidTensorView<complex<double>, 2>;


  template<typename T>
  using HostVector = FluidTensorView<T,1>;

  template<typename T>
  using HostMatrix = FluidTensorView<T,2>;

} // namespace fluid
