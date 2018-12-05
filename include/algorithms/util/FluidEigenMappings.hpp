#pragma once

#include "../../data/FluidTensor.hpp"
#include <Eigen/Core>
#include <Eigen/Eigen>

namespace fluid {
namespace algorithm {

using Eigen::PlainObjectBase;
using Eigen::Map;
using Eigen::Stride;
using Eigen::Array;
using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::RowMajor;
using Eigen::StorageOptions;
using Eigen::AlignmentType;
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

/**
 Utility functions for converting between FluidTensorView and Eigen wrappers around raw poiniters
 **/

namespace fluid {
namespace algorithm {
namespace _impl {

  using Eigen::PlainObjectBase;
  using Eigen::Map;
  using Eigen::Stride;

  using Eigen::Dynamic;
  using Eigen::RowMajor;
  using Eigen::ColMajor;
  using Eigen::AlignmentType;

  /**
   Convert an Eigen Matrix or Array to a FluidTensorView
   **/
  template<typename Derived>
  auto asFluid(PlainObjectBase<Derived> &a)
  -> FluidTensorView<typename PlainObjectBase<Derived>::Scalar, PlainObjectBase<Derived>::IsVectorAtCompileTime ? 1:2>
  {
    constexpr size_t N = PlainObjectBase<Derived>::IsVectorAtCompileTime ? 1 : 2;

    if(N==2)
    {
      if(a.Options == ColMajor)
      {
        //Respect the colmajorness of an eigen type
        auto slice =
        fluid::FluidTensorSlice<N>(0, {static_cast<size_t>(a.rows()),static_cast<size_t>(a.cols())},{1,static_cast<size_t>(a.rows())});
        return {slice, a.data()};
      }
      return {a.data(),0,a.rows(),a.cols()};
    }
    else
      return {a.data(),0,a.rows()};
  }

  /**
   Convert an lvalue FluidTensorView to an Eigen Matrix or Array map (you need to say which as a template parmeter, e.g. makeWrapper<Matrix>(myView)
  **/
  template<template <typename, int, int, int, int, int> class EigenType, typename T, size_t N, template <typename, size_t> class F>
  auto asFluid(F<T,N>& a)
  -> Map<EigenType<T,  Dynamic, Dynamic, RowMajor, Dynamic, Dynamic> , Eigen::AlignmentType::Unaligned, Stride<Dynamic, Dynamic>>
  {
    static_assert(N<3, "Can't convert to Eigen types with more than two dimensions" );

    if(N==2)
    {
      return {a.data(), static_cast<Eigen::Index>(a.rows()), static_cast<Eigen::Index>(a.cols()), Stride<Dynamic,Dynamic>(a.descriptor().strides[0],a.descriptor().strides[1])};
    }
    else
    {
      return {a.data(), static_cast<Eigen::Index>(a.rows()), 1, Stride<Dynamic,Dynamic>(a.descriptor().strides[0],1)};
    }
  }

  /**
   Convert an const lvalue FluidTensorView to a const Eigen Matrix or Array map (you need to say which as a template parmeter, e.g. makeWrapper<Matrix>(myConstView)
   **/
  template<template <typename, int, int, int, int, int> class EigenType, typename T, size_t N, template <typename, size_t> class F>
  auto asEigen(const F<T,N>& a)
  -> Map<const EigenType<T,  Dynamic, Dynamic, RowMajor, Dynamic, Dynamic> , Eigen::AlignmentType::Unaligned, Stride<Dynamic, Dynamic>>
  {
    static_assert(N<3, "Can't convert to Eigen types with more than two dimensions" );

    if(N==2)
    {
      return {a.data(), static_cast<Eigen::Index>(a.rows()), static_cast<Eigen::Index>(a.cols()), Stride<Dynamic,Dynamic>(a.descriptor().strides[0],a.descriptor().strides[1])};
    }
    else
    {
      return {a.data(), static_cast<Eigen::Index>(a.rows()), 1, Stride<Dynamic,Dynamic>(a.descriptor().strides[0],1)};
    }
  }


  /**
   Convert an rvalue FluidTensorView to a Eigen Matrix or Array map (you need to say which as a template parmeter, e.g. makeWrapper<Matrix>(myView.row(0))
   **/
  template<template <typename, int, int, int, int, int> class EigenType, typename T, size_t N>
  auto asEigen(FluidTensorView<T,N>&& a) //restrict this to FluidTensorView because passing in an rvalue FluidTensor would be silly
  -> Map<EigenType<T,  Dynamic, Dynamic, RowMajor, Dynamic, Dynamic> , Eigen::AlignmentType::Unaligned, Stride<Dynamic, Dynamic>>
  {
    return asEigen<EigenType>(a);
  }

  /**
   Convert a const rvalue FluidTensorView to a const Eigen Matrix or Array map (you need to say which as a template parmeter, e.g. makeWrapper<Matrix>(myCOnstView.row(0))
   **/
  template<template <typename, int, int, int, int, int> class EigenType, typename T, size_t N>
  auto asEigen(const FluidTensorView<T,N>&& a) //restrict this to FluidTensorView because passing in an rvalue FluidTensor would be silly
  -> Map<const EigenType<T,  Dynamic, Dynamic, RowMajor, Dynamic, Dynamic> , Eigen::AlignmentType::Unaligned, Stride<Dynamic, Dynamic>>
  {
    return asEigen<EigenType>(a);
  }
}
}
}
