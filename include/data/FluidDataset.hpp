#pragma once

#include "data/FluidTensor.hpp"
#include "data/TensorTypes.hpp"
#include <string>
#include <unordered_map>

namespace fluid {

template <typename idType, typename dataType, typename targetType, size_t N>
class FluidDataset {

public:
  template <typename... Dims,
            typename = std::enable_if_t<isIndexSequence<Dims...>()>>
  FluidDataset(Dims... dims) : mData(0, dims...), mDim(dims...) {
    static_assert(sizeof...(dims) == N, "Number of dimensions doesn't match");
  }

  bool add(idType id, FluidTensorView<dataType, N> point, targetType target = targetType()) {
    assert(sameExtents(mDim, point.descriptor()));
    std::cout<<"ds add "<<id<<":"<<target<<std::endl;
    intptr_t pos = mData.rows();
    auto result = mIndex.insert({id, pos});
    if (!result.second)
      return false;
    mTargets.resize(mTargets.rows() + 1);
    mTargets(mTargets.rows() - 1) = target;
    mData.resizeDim(0, 1);
    mData.row(mData.rows() - 1) = point;
    std::cout<<mData.rows()<<std::endl;
    return true;
  }


  bool get(idType id, FluidTensorView<dataType, N> point) {
    auto pos = mIndex.find(id);
    if (pos == mIndex.end())
      return false;
    point = mData.row(pos->second);
    return true;
  }

  bool update(idType id, FluidTensorView<dataType, N> point) {
      auto pos = mIndex.find(id);
      if (pos == mIndex.end())
        return false;
      else
        mData.row(pos->second) = point;
      return true;
  }

  bool remove(idType id) {
      auto pos = mIndex.find(id);
      if (pos == mIndex.end())
        return false;
      else {
        mTargets.deleteRow(pos->second);
        mData.deleteRow(pos->second);
        mIndex.erase(id);
      }
      return true;
  }

  size_t pointSize() const { return mDim.size; }
  size_t size() const { return mTargets.size(); }
  void print() const {
     std::cout << mData << std::endl;
     std::cout << mTargets << std::endl;

   }

  const FluidTensorView<dataType, N + 1> getData() const { return mData; }

  const FluidTensorView<targetType, 1> getTargets() const { return mTargets; }
  const FluidTensorView<idType, 1> getIds() const { return mIndex; }

private:
  mutable FluidTensor<targetType, 1> mTargets;
  mutable std::unordered_map<idType, intptr_t> mIndex;
  mutable FluidTensor<dataType, N + 1> mData;
  FluidTensorSlice<N> mDim;
};
} // namespace fluid
