/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
#pragma once

#include "FluidTensor.hpp"
#include <Eigen/Core> 
#include <complex>

namespace fluid {

using RealMatrix = FluidTensor<double, 2>;
using ComplexMatrix = FluidTensor<std::complex<double>, 2>;
using RealVector = FluidTensor<double, 1>;
using ComplexVector = FluidTensor<std::complex<double>, 1>;

using RealMatrixView = FluidTensorView<double, 2>;
using ComplexMatrixView = FluidTensorView<std::complex<double>, 2>;
using RealVectorView = FluidTensorView<double, 1>;
using ComplexVectorView = FluidTensorView<std::complex<double>, 1>;

using InputRealMatrixView = FluidTensorView<const double, 2>;
using InputComplexMatrixView = FluidTensorView<const std::complex<double>, 2>;
using InputRealVectorView = FluidTensorView<const double, 1>;
using InputComplexVectorView = FluidTensorView<const std::complex<double>, 1>;

using ArrayXXidx = Eigen::Array<fluid::index, Eigen::Dynamic, Eigen::Dynamic>; 
using ArrayXidx = Eigen::Array<fluid::index, Eigen::Dynamic, 1>; 
} // namespace fluid
