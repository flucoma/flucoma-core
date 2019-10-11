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

  FluidDataset(FluidTensor<idType, 1> ids, FluidTensor<dataType, N + 1> points,
    FluidTensor<targetType, 1> targets) {
    assert(ids.rows() == points.rows() && ids.rows() == targets.rows());
    mData = points;
    mTargets = targets;
    mDim = mData.cols();
    mIds = ids;
    for(int i = 0; i < ids.size();i++){
      mIndex.insert({ids[i],i});
    }
  }


  bool add(idType id, FluidTensorView<dataType, N> point, targetType target = targetType()) {
    assert(sameExtents(mDim, point.descriptor()));
    intptr_t pos = mData.rows();
    auto result = mIndex.insert({id, pos});
    if (!result.second)
      return false;
    mTargets.resize(mTargets.rows() + 1);
    mTargets(mTargets.rows() - 1) = target;
    mData.resizeDim(0, 1);
    mData.row(mData.rows() - 1) = point;
    mIds.resizeDim(0, 1);
    mIds(mIds.rows() - 1) = id;
    return true;
  }


  bool get(idType id, FluidTensorView<dataType, N> point) {
    auto pos = mIndex.find(id);
    if (pos == mIndex.end())
      return false;
    point = mData.row(pos->second);
    return true;
  }

  targetType getTarget(idType id) {
    auto pos = mIndex.find(id);
    if (pos == mIndex.end())
      return targetType();
    targetType target = mTargets(pos->second);
    return target;
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
  const FluidTensorView<idType, 1> getIds() const { return mIds; }

private:
  mutable FluidTensor<targetType, 1> mTargets;
  mutable std::unordered_map<idType, intptr_t> mIndex;
  mutable FluidTensor<idType, 1> mIds;
  mutable FluidTensor<dataType, N + 1> mData;
  FluidTensorSlice<N> mDim;
};
} // namespace fluid
