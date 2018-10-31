#pragma once

#include <Eigen/Eigen>

namespace fluid {

using Eigen::Dynamic;
using Eigen::Matrix;

template <typename Scalar>
Matrix<Scalar, Dynamic, Dynamic>
toeplitz(const Matrix<Scalar, Dynamic, 1> &vec) {
  size_t size = vec.size();
  Matrix<Scalar, Dynamic, Dynamic> mat(size, size);

  for (auto i = 0; i < size; i++) {
    for (auto j = 0; j < i; j++)
      mat(j, i) = vec(i - j);
    for (auto j = i; j < size; j++)
      mat(j, i) = vec(j - i);
  }

  return mat;
}

}; // namespace fluid
