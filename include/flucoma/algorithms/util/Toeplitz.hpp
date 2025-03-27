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

#include "../../data/FluidIndex.hpp"
#include <Eigen/Core>

namespace fluid {
namespace algorithm {

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
toeplitz(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& vec)
{
  index size = vec.size();

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> mat(size, size);

  for (auto i = 0; i < size; i++)
  {
    for (auto j = 0; j < i; j++) mat(j, i) = vec(i - j);
    for (auto j = i; j < size; j++) mat(j, i) = vec(j - i);
  }

  return mat;
}

} // namespace algorithm
} // namespace fluid
