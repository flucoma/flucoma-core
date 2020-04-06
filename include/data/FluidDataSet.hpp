#pragma once

#include "data/FluidTensor.hpp"
#include "data/TensorTypes.hpp"
#include <string>
#include <unordered_map>

namespace fluid {

template <typename idType, typename dataType, size_t N>
class FluidDataSet {

public:
  template <typename... Dims,
            typename = std::enable_if_t<isIndexSequence<Dims...>()>>
  FluidDataSet(Dims... dims) : mData(0, dims...), mDim(dims...) {
    static_assert(sizeof...(dims) == N, "Number of dimensions doesn't match");
  }

  FluidDataSet(FluidTensor<idType, 1> ids, FluidTensor<dataType, N + 1> points) {
    assert(ids.rows() == points.rows());
    mData = points;
    mDim = mData.cols();
    mIds = ids;
    for(int i = 0; i < ids.size();i++){
      mIndex.insert({ids[i],i});
    }
  }


  bool add(idType id, FluidTensorView<dataType, N> point) {
    assert(sameExtents(mDim, point.descriptor()));
    intptr_t pos = mData.rows();
    auto result = mIndex.insert({id, pos});
    if (!result.second)
      return false;
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
      if (pos == mIndex.end()){
        return false;
      }
      else {
        auto current = pos->second;
        mData.deleteRow(current);
        mIds.deleteRow(current);
        mIndex.erase(id);
        for( auto& point : mIndex)
          if(point.second > current) point.second--;
      }
      return true;
  }

  size_t pointSize() const { return mDim.size; }
  size_t size() const { return mIds.size(); }
  void print() const {
    if(size()>0){
     std::cout << mData << std::endl;
     std::cout << mIds << std::endl;
   }
   }

  const FluidTensorView<dataType, N + 1> getData() const { return mData; }
  const FluidTensorView<idType, 1> getIds() const { return mIds; }

private:
  mutable std::unordered_map<idType, intptr_t> mIndex;
  mutable FluidTensor<idType, 1> mIds;
  mutable FluidTensor<dataType, N + 1> mData;
  FluidTensorSlice<N> mDim;
};
} // namespace fluid
