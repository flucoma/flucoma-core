#pragma once

#include "data/FluidTensor.hpp"
#include "data/TensorTypes.hpp"
#include <string>

namespace fluid {

template <typename T, typename U, size_t N> class FluidDataset
{

public:
  template <typename... Dims,
            typename = std::enable_if_t<isIndexSequence<Dims...>()>>
  FluidDataset(Dims... dims) : mData(0, dims...), mDim(dims...)
  {
    static_assert(sizeof...(dims) == N, "Number of dimensions doesn't match");
  }

  bool add(U label, FluidTensorView<T, N> point)
  {
    assert(sameExtents(mDim, point.descriptor()));
    auto pos = std::find(mLabels.begin(), mLabels.end(), label);
    if (pos != mLabels.end())
      return false;
    mLabels.resize(mLabels.rows() + 1);
    mLabels(mLabels.rows() - 1) = label;
    mData.resizeDim(0, 1);
    mData.row(mData.rows() - 1) = point;
    return true;
  }

  bool get(U label, FluidTensorView<T, N> point)
  {
    auto pos = std::find(mLabels.begin(), mLabels.end(), label);
    if (pos == mLabels.end())
      return false;
    point = mData.row(std::distance(mLabels.begin(), pos));
    return true;
  }

  bool update(U label, FluidTensorView<T, N> point)
  {
    auto pos = std::find(mLabels.begin(), mLabels.end(), label);
    if (pos == mLabels.end())
      return false;
    else
      mData.row(std::distance(mLabels.begin(), pos)) = point;
    return true;
  }

  bool remove(U label)
  {
    auto pos = std::find(mLabels.begin(), mLabels.end(), label);
    if (pos == mLabels.end())
      return false;
    else {
      mLabels.deleteRow(std::distance(mLabels.begin(), pos));
      mData.deleteRow(std::distance(mLabels.begin(), pos));
    }
    return true;
  }

  size_t pointSize() { return mDim.size; }
  void print() { std::cout << mData << std::endl; }

private:
  FluidTensor<U, 1> mLabels;
  FluidTensor<T, N + 1> mData;
  FluidTensorSlice<N> mDim;
};
} // namespace fluid
