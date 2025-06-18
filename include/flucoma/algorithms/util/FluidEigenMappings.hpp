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

#include "../../data/FluidMemory.hpp"
#include "../../data/FluidTensor.hpp"
#include <Eigen/Core>
#include <algorithm>

/// converting between FluidTensorView and Eigen wrappers around raw poiniters

namespace fluid {
namespace algorithm {

using Eigen::Map;
using Eigen::PlainObjectBase;
using Eigen::Stride;

using Eigen::AlignmentType;
using Eigen::ColMajor;
using Eigen::Dynamic;
using Eigen::RowMajor;

namespace _impl {


template <typename EigenType>
auto asFluid(ScopedEigenMap<EigenType>& a)
    -> FluidTensorView<typename EigenType::Scalar,
                       (EigenType::IsVectorAtCompileTime ? 1 : 2)>
{
  constexpr size_t N = EigenType::IsVectorAtCompileTime ? 1 : 2;

  if constexpr (N == 2)
  {
    if (static_cast<int>(a.Options) == static_cast<int>(ColMajor))
    {
      // Respect the colmajorness of an eigen type
      auto slice = FluidTensorSlice<N>(0, {a.cols(), a.rows()});
      return {slice.transpose(), a.data()};
    }
    return {a.data(), 0, a.rows(), a.cols()};
  }
  else
    return {a.data(), 0, a.rows()};
}

/// Eigen Matrix<T>/Array<T> -> FluidTensorView<T>
template <typename Derived>
auto asFluid(PlainObjectBase<Derived>& a)
    -> FluidTensorView<typename PlainObjectBase<Derived>::Scalar,
                       (PlainObjectBase<Derived>::IsVectorAtCompileTime ? 1
                                                                        : 2)>
{
  constexpr size_t N = PlainObjectBase<Derived>::IsVectorAtCompileTime ? 1 : 2;

  if constexpr (N == 2)
  {
    if (static_cast<int>(a.Options) == static_cast<int>(ColMajor))
    {
      // Respect the colmajorness of an eigen type
      auto slice = FluidTensorSlice<N>(0, {a.cols(), a.rows()});
      return {slice.transpose(), a.data()};
    }
    return {a.data(), 0, a.rows(), a.cols()};
  }
  else
    return {a.data(), 0, a.rows()};
}

/// const Eigen::Matrix/Array -> FluidTensorView<const T>
template <typename Derived>
auto asFluid(const PlainObjectBase<Derived>& a)
    -> FluidTensorView<const typename PlainObjectBase<Derived>::Scalar,
                       (PlainObjectBase<Derived>::IsVectorAtCompileTime ? 1
                                                                        : 2)>
{
  constexpr size_t N = PlainObjectBase<Derived>::IsVectorAtCompileTime ? 1 : 2;

  if constexpr (N == 2)
  {
    if (static_cast<int>(a.Options) == static_cast<int>(ColMajor))
    {
      // Respect the colmajorness of an eigen type
      auto slice = FluidTensorSlice<N>(0, {a.cols(), a.rows()});
      return {slice.transpose(), a.data()};
    }
    return {a.data(), 0, a.rows(), a.cols()};
  }
  else
    return {a.data(), 0, a.rows()};
}


/// lvalue FluidTensor<T> / FluidTensorView<T> -> Matrix/Array<T> (say which as
/// template param) e.g. asEigen<Matrix>(myView)

template <template <typename, int, int, int, int, int> class EigenType,
          typename T, size_t N>
auto asEigen(FluidTensor<T, N>& a)
    -> Map<EigenType<T, Dynamic, Dynamic, RowMajor, Dynamic, Dynamic>,
           Eigen::AlignmentType::Unaligned, Stride<Dynamic, Dynamic>>
{
  static_assert(N < 3,
                "Can't convert to Eigen types with more than two dimensions");

  if (N == 2)
  {
    return {a.data(), static_cast<Eigen::Index>(a.rows()),
            static_cast<Eigen::Index>(a.cols()),
            Stride<Dynamic, Dynamic>(a.descriptor().strides[0],
                                     a.descriptor().strides[1])};
  }
  else
  {
    return {a.data(), static_cast<Eigen::Index>(a.rows()), 1,
            Stride<Dynamic, Dynamic>(a.descriptor().strides[0], 1)};
  }
}

/// lvalue const FluidTensor<T> -> const Matrix<T>/ const Array<T>
template <template <typename, int, int, int, int, int> class EigenType,
          typename T, size_t N>
auto asEigen(const FluidTensor<T, N>& a)
    -> Map<const EigenType<T, Dynamic, Dynamic, RowMajor, Dynamic, Dynamic>,
           Eigen::AlignmentType::Unaligned, Stride<Dynamic, Dynamic>>
{
  static_assert(N < 3,
                "Can't convert to Eigen types with more than two dimensions");

  if (N == 2)
  {
    return {a.data(), static_cast<Eigen::Index>(a.rows()),
            static_cast<Eigen::Index>(a.cols()),
            Stride<Dynamic, Dynamic>(a.descriptor().strides[0],
                                     a.descriptor().strides[1])};
  }
  else
  {
    return {a.data(), static_cast<Eigen::Index>(a.rows()), 1,
            Stride<Dynamic, Dynamic>(a.descriptor().strides[0], 1)};
  }
}


/// lvalue FluidTensorView<T> ->  Matrix / Array
template <template <typename, int, int, int, int, int> class EigenType,
          typename T, size_t N>
auto asEigen(const FluidTensorView<T, N>& a) -> Map<
    EigenType<std::decay_t<T>, Dynamic, Dynamic, RowMajor, Dynamic, Dynamic>,
    Eigen::AlignmentType::Unaligned, Stride<Dynamic, Dynamic>>
{
  static_assert(N < 3,
                "Can't convert to Eigen types with more than two dimensions");

  if (N == 2)
  {
    return {a.data(), static_cast<Eigen::Index>(a.rows()),
            static_cast<Eigen::Index>(a.cols()),
            Stride<Dynamic, Dynamic>(a.descriptor().strides[0],
                                     a.descriptor().strides[1])};
  }
  else
  {
    return {a.data(), static_cast<Eigen::Index>(a.rows()), 1,
            Stride<Dynamic, Dynamic>(a.descriptor().strides[0], 1)};
  }
}

/// lvalue FluidTensorView<const T> ->  const Matrix / Array
template <template <typename, int, int, int, int, int> class EigenType,
          typename T, size_t N>
auto asEigen(const FluidTensorView<const T, N>& a)
    -> Map<const EigenType<std::decay_t<T>, Dynamic, Dynamic, RowMajor, Dynamic,
                           Dynamic>,
           Eigen::AlignmentType::Unaligned, Stride<Dynamic, Dynamic>>
{
  static_assert(N < 3,
                "Can't convert to Eigen types with more than two dimensions");

  if (N == 2)
  {
    return {a.data(), static_cast<Eigen::Index>(a.rows()),
            static_cast<Eigen::Index>(a.cols()),
            Stride<Dynamic, Dynamic>(a.descriptor().strides[0],
                                     a.descriptor().strides[1])};
  }
  else
  {
    return {a.data(), static_cast<Eigen::Index>(a.rows()), 1,
            Stride<Dynamic, Dynamic>(a.descriptor().strides[0], 1)};
  }
}

/// rvalue FluidTensorView<T> ->  Matrix / Array
template <template <typename, int, int, int, int, int> class EigenType,
          typename T, size_t N>
auto asEigen(const FluidTensorView<T, N>&&
                 a) // restrict this to FluidTensorView because
                    // passing in an rvalue FluidTensor would be silly
    -> Map<EigenType<T, Dynamic, Dynamic, RowMajor, Dynamic, Dynamic>,
           Eigen::AlignmentType::Unaligned, Stride<Dynamic, Dynamic>>
{
  return asEigen<EigenType>(a);
}

/// rvalue FluidTensorView<const T> ->  const Matrix / Array
template <template <typename, int, int, int, int, int> class EigenType,
          typename T, size_t N>
auto asEigen(const FluidTensorView<const T, N>&&
                 a) // restrict this to FluidTensorView because passing in an
                    // rvalue FluidTensor would be silly
    -> Map<const EigenType<T, Dynamic, Dynamic, RowMajor, Dynamic, Dynamic>,
           Eigen::AlignmentType::Unaligned, Stride<Dynamic, Dynamic>>
{
  return asEigen<EigenType>(a);
}

} // namespace _impl

template <template <typename, int, int, int, int, int> class EigenType>
using FluidEigenMap =
    Eigen::Map<EigenType<double, Dynamic, Dynamic, RowMajor, Dynamic, Dynamic>,
               Eigen::AlignmentType::Unaligned, Stride<Dynamic, Dynamic>>;

} // namespace algorithm
} // namespace fluid
