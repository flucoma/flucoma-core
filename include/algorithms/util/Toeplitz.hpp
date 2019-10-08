#pragma once

#include <Eigen/Core>

namespace fluid {
namespace algorithm {

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
toeplitz(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &vec) {
  size_t size = vec.size();
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> mat(size, size);

  for (auto i = 0; i < size; i++) {
    for (auto j = 0; j < i; j++)
      mat(j, i) = vec(i - j);
    for (auto j = i; j < size; j++)
      mat(j, i) = vec(j - i);
  }

  return mat;
}

}
}; // namespace fluid
