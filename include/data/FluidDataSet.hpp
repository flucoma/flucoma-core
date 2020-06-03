#pragma once

#include "data/FluidIndex.hpp"
#include "data/FluidTensor.hpp"
#include "data/TensorTypes.hpp"
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>

namespace fluid {

template <typename idType, typename dataType, index N> class FluidDataSet {

public:

  template <typename... Dims,
            typename = std::enable_if_t<isIndexSequence<Dims...>()>>
  FluidDataSet(Dims... dims) : mData(0, dims...), mDim(dims...) {
    static_assert(sizeof...(dims) == N, "Number of dimensions doesn't match");
  }

  explicit FluidDataSet() = default;
  ~FluidDataSet() = default;

  template <typename... Dims,
            typename = std::enable_if_t<isIndexSequence<Dims...>()>>
  bool resize(Dims... dims) {
    static_assert(sizeof...(dims) == N, "Number of dimensions doesn't match");
    if (size() == 0) {
      mData = FluidTensor<dataType, N + 1>(0, dims...);
      mDim = FluidTensorSlice<N>(dims...);
      return true;
    } else {
      return false;
    }
  }

  FluidDataSet(FluidTensor<idType, 1> ids,
               FluidTensor<dataType, N + 1> points) {
    assert(ids.rows() == points.rows());
    mData = points;
    mDim = mData.cols();
    mIds = ids;
    for (index i = 0; i < ids.size(); i++) {
      mIndex.insert({ids[i], i});
    }
  }

  bool add(idType id, FluidTensorView<dataType, N> point) {
    assert(sameExtents(mDim, point.descriptor()));
    index pos = mData.rows();
    auto result = mIndex.insert({id, pos});
    if (!result.second)
      return false;
    mData.resizeDim(0, 1);
    mData.row(mData.rows() - 1) = point;
    mIds.resizeDim(0, 1);
    mIds(mIds.rows() - 1) = id;
    return true;
  }

  bool get(idType id, FluidTensorView<dataType, N> point) const {
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
    if (pos == mIndex.end()) {
      return false;
    } else {
      auto current = pos->second;
      mData.deleteRow(current);
      mIds.deleteRow(current);
      mIndex.erase(id);
      for (auto &point : mIndex)
        if (point.second > current)
          point.second--;
    }
    return true;
  }

  index pointSize() const { return mDim.size; }
  index dims() const { return mDim.size; }
  index size() const { return mIds.size(); }

  std::string printRow(FluidTensorView<dataType, N> row, index maxCols) const {
    using namespace std;
    ostringstream result;
    if (row.size() < maxCols) {
      for (index c = 0; c < row.size(); c++) {
        result << setw(10) << setprecision(5) << row(c);
      }
    } else {
      for (index c = 0; c < maxCols / 2; c++) {
        result << setw(10) << setprecision(5) << row(c);
      }
      result << setw(10) << "...";
      for (index c = maxCols / 2; c > 0; c--) {
        result << setw(10) << setprecision(5) << row(row.size() - c);
      }
    }
    return result.str();
  }

  std::string print(index maxRows = 6, index maxCols = 6) const {
    using namespace std;
    if (size() == 0)
      return "{}";
    ostringstream result;
    result << endl << "rows: " << size() << " cols: " << pointSize() << endl;
    if (size() < maxRows) {
      for (index r = 0; r < size(); r++) {
        result << mIds(r) << " " << printRow(mData.row(r), maxCols)
               << std::endl;
      }
    } else {
      for (index r = 0; r < maxRows / 2; r++) {
        result << mIds(r) << " " << printRow(mData.row(r), maxCols)
               << std::endl;
      }
      result << setw(10) << "..." << std::endl;
      for (index r = maxRows / 2; r > 0; r--) {
        result << mIds(size() - r) << " "
               << printRow(mData.row(size() - r), maxCols) << std::endl;
      }
    }
    return result.str();
  }
  const FluidTensorView<dataType, N + 1> getData() const { return mData; }
  const FluidTensorView<idType, 1> getIds() const { return mIds; }

private:
  mutable std::unordered_map<idType, index> mIndex;
  mutable FluidTensor<idType, 1> mIds;
  mutable FluidTensor<dataType, N + 1> mData;
  FluidTensorSlice<N> mDim;
};
} // namespace fluid
