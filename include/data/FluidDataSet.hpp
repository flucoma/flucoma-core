#pragma once

#include "data/FluidIndex.hpp"
#include "data/FluidTensor.hpp"
#include "data/TensorTypes.hpp"
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>

namespace fluid {

template <typename idType, typename dataType, index N>
class FluidDataSet
{

public:
  explicit FluidDataSet() = default;
  ~FluidDataSet() = default;

  // Construct from list of dimensions for each data point,
  // e.g. FluidDataSet(2, 3) is a dataset of 2x3 tensors
  template <typename... Dims,
            typename = std::enable_if_t<isIndexSequence<Dims...>()>>
  FluidDataSet(Dims... dims) : mData(0, dims...), mDim(dims...)
  {
    static_assert(sizeof...(dims) == N, "Number of dimensions doesn't match");
  }

  // Construct from existing tensors of ids and data points
  FluidDataSet(FluidTensorView<const idType, 1>       ids,
               FluidTensorView<const dataType, N + 1> points)
      : mIds(ids), mData(points)
  {
    initFromData();
  }

  // Construct from existing tensors of ids and data points
  // (from convertible type for data, typically float -> double)
  template <typename U, typename T = dataType>
  FluidDataSet(FluidTensorView<const idType, 1> ids,
               FluidTensorView<const U, N + 1>  points,
               std::enable_if_t<std::is_convertible<U, T>::value>* = nullptr)
      : mIds(ids), mData(points)
  {
    initFromData();
  }

  // Resize data point layout (if empty)
  template <typename... Dims,
            typename = std::enable_if_t<isIndexSequence<Dims...>()>>
  bool resize(Dims... dims)
  {
    static_assert(sizeof...(dims) == N, "Number of dimensions doesn't match");
    if (size() == 0)
    {
      mData = FluidTensor<dataType, N + 1>(0, dims...);
      mDim = FluidTensorSlice<N>(dims...);
      return true;
    }
    else
    {
      return false;
    }
  }

  bool add(idType const& id, FluidTensorView<dataType, N> point)
  {
    assert(sameExtents(mDim, point.descriptor()));
    index pos = mData.rows();
    auto  result = mIndex.insert({id, pos});
    if (!result.second) return false;
    mData.resizeDim(0, 1);
    mData.row(mData.rows() - 1) <<= point;
    mIds.resizeDim(0, 1);
    mIds(mIds.rows() - 1) = id;
    return true;
  }

  bool get(idType const& id, FluidTensorView<dataType, N> point) const
  {
    auto pos = mIndex.find(id);
    if (pos == mIndex.end()) return false;
    point <<= mData.row(pos->second);
    return true;
  }

  FluidTensorView<const dataType, N> get(idType const& id) const
  {
    auto pos = mIndex.find(id);
    return pos != mIndex.end()
               ? mData.row(pos->second)
               : FluidTensorView<const dataType, N>{nullptr, 0, 0};
  }

  index getIndex(idType const& id) const
  {
    auto pos = mIndex.find(id);
    if (pos == mIndex.end())
      return -1;
    else
      return pos->second;
  }

  bool update(idType const& id, FluidTensorView<dataType, N> point)
  {
    auto pos = mIndex.find(id);
    if (pos == mIndex.end())
      return false;
    else
      mData.row(pos->second) <<= point;
    return true;
  }

  bool remove(idType const& id)
  {
    auto pos = mIndex.find(id);
    if (pos == mIndex.end()) { return false; }
    else
    {
      auto current = pos->second;
      mData.deleteRow(current);
      mIds.deleteRow(current);
      mIndex.erase(id);
      for (auto& point : mIndex)
        if (point.second > current) point.second--;
    }
    return true;
  }

  FluidTensorView<dataType, N + 1>       getData() { return mData; }
  FluidTensorView<idType, 1>             getIds() { return mIds; }
  FluidTensorView<const dataType, N + 1> getData() const { return mData; }
  FluidTensorView<const idType, 1>       getIds() const { return mIds; }

  index pointSize() const { return mDim.size; }
  index dims() const { return mDim.size; }
  index size() const { return mIds.size(); }
  bool  initialized() const { return (size() > 0); }

  std::string printRow(FluidTensorView<const dataType, N> row,
                       index                              maxCols) const
  {
    using namespace std;
    ostringstream result;
    if (row.size() < maxCols)
    {
      for (index c = 0; c < row.size(); c++)
      {
        result << setw(10) << setprecision(5) << row(c);
      }
    }
    else
    {
      for (index c = 0; c < maxCols / 2; c++)
      {
        result << setw(10) << setprecision(5) << row(c);
      }
      result << setw(10) << "...";
      for (index c = maxCols / 2; c > 0; c--)
      {
        result << setw(10) << setprecision(5) << row(row.size() - c);
      }
    }
    return result.str();
  }

  std::string print(index maxRows = 6, index maxCols = 6) const
  {
    using namespace std;
    if (size() == 0) return "{}";
    ostringstream result;
    result << endl << "rows: " << size() << " cols: " << pointSize() << endl;
    if (size() < maxRows)
    {
      for (index r = 0; r < size(); r++)
      {
        result << mIds(r) << " " << printRow(mData.row(r), maxCols)
               << std::endl;
      }
    }
    else
    {
      for (index r = 0; r < maxRows / 2; r++)
      {
        result << mIds(r) << " " << printRow(mData.row(r), maxCols)
               << std::endl;
      }
      result << setw(10) << "..." << std::endl;
      for (index r = maxRows / 2; r > 0; r--)
      {
        result << mIds(size() - r) << " "
               << printRow(mData.row(size() - r), maxCols) << std::endl;
      }
    }
    return result.str();
  }

private:
  void initFromData()
  {
    assert(mIds.rows() == mData.rows());
    mDim = mData.cols();
    for (index i = 0; i < mIds.size(); i++) { mIndex.insert({mIds[i], i}); }
  }

  std::unordered_map<idType, index> mIndex;
  FluidTensor<idType, 1>            mIds;
  FluidTensor<dataType, N + 1>      mData;
  FluidTensorSlice<N>               mDim;
};
} // namespace fluid
