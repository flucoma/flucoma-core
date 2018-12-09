#pragma once

#include "FluidTensor.hpp"
#include <complex>
#include <type_traits>

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

  template<class Matrix, typename T=typename Matrix::type, size_t N=Matrix::order>
  using IsView = std::is_same< Matrix, FluidTensorView<T,N>>;
  
  template<class Matrix, class = std::enable_if_t<IsView<Matrix>::value>>
  using Data = FluidTensor<typename Matrix::type, Matrix::order>;
} // namespace fluid
