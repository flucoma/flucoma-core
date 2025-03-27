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

#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidDataSet.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <set>
#include <string>

namespace fluid {
namespace algorithm {

class DataSetQuery
{

public:
  using string = std::string;

  using DataSet = FluidDataSet<string, double, 1>;

  struct Condition
  {
    index  column;
    index  comparison;
    double value;

    bool test(InputRealVectorView point)
    {
      using namespace std;
      switch (comparison)
      {
      case 0: return point(column) == value;
      case 1: return point(column) != value;
      case 2: return point(column) < value;
      case 3: return point(column) <= value;
      case 4: return point(column) > value;
      case 5: return point(column) >= value;
      }
      return false;
    }
  };

  DataSetQuery()
  {
    mComparisons = std::vector<std::string>{"==", "!=", "<", "<=", ">", ">="};
  }

  void addColumn(index col) { mColumns.emplace(col); }

  void addRange(index from, index count)
  {
    for (index i = from; i < from + count; i++) { mColumns.emplace(i); }
  }

  index numColumns() { return asSigned(mColumns.size()); }

  bool hasAndConditions() { return (mAndConditions.size() > 0); }

  index maxColumn() { return mColumns.empty() ? 0 : *mColumns.rbegin(); }

  bool addCondition(
      index column, string comparison, double value, bool conjunction)
  {
    auto pos = std::find(mComparisons.begin(), mComparisons.end(), comparison);
    if (pos == mComparisons.end()) return false;
    index i = std::distance(mComparisons.begin(), pos);
    if (conjunction)
      mAndConditions.emplace_back(Condition{column, i, value});
    else
      mOrConditions.emplace_back(Condition{column, i, value});
    return true;
  }

  void process(const DataSet& input, DataSet& current, DataSet& output)
  {
    auto data = input.getData();
    auto ids = input.getIds();
    mTmpPoint = RealVector(current.pointSize() + asSigned(mColumns.size()));
    index limit = mLimit == 0 ? input.size() : mLimit;
    index count = 0;
    for (index i = 0; i < input.size() && count < limit; i++)
    {
      bool matchesAllAnd = true;
      auto point = data.row(i);
      for (index j = 0; j < asSigned(mAndConditions.size()); j++)
        if (!mAndConditions[asUnsigned(j)].test(point)) matchesAllAnd = false;
      if (matchesAllAnd)
      {
        addRow(ids(i), point, current, output);
        count++;
        continue;
      }
      else
      {
        bool matchesAnyOr = false;
        for (index k = 0; k < asSigned(mOrConditions.size()); k++)
          if (mOrConditions[asUnsigned(k)].test(point)) matchesAnyOr = true;
        if (matchesAnyOr)
        {
          addRow(ids(i), point, current, output);
          count++;
        }
      }
    }
  }

  void print() const {}

  void limit(index rows) { mLimit = rows; }
  void clear()
  {
    mColumns.clear();
    mAndConditions.clear();
    mOrConditions.clear();
    mLimit = 0;
  }

private:
  void addRow(
      string id, InputRealVectorView point, const DataSet& current, DataSet& out)
  {
    mTmpPoint.fill(0);
    index currentSize = current.pointSize();
    bool  shouldAdd = true;
    if (currentSize > 0)
    {
      RealVectorView currentData = mTmpPoint(Slice(0, currentSize));
      shouldAdd = current.get(id, currentData);
    }
    if (shouldAdd)
    {
      for (auto c : mColumns) mTmpPoint(currentSize++) = point(c);
      out.add(id, mTmpPoint);
    }
  }

  index                    mLimit{0};
  std::set<index>          mColumns;
  RealVector               mTmpPoint;
  std::vector<std::string> mComparisons;
  std::vector<Condition>   mAndConditions;
  std::vector<Condition>   mOrConditions;
};
} // namespace algorithm
} // namespace fluid
