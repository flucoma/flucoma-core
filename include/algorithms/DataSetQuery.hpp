#pragma once

#include "algorithms/util/FluidEigenMappings.hpp"
#include "data/FluidDataSet.hpp"
#include "data/FluidTensor.hpp"
#include "data/TensorTypes.hpp"
#include "data/FluidIndex.hpp"
#include <Eigen/Core>
#include <set>
#include <string>

namespace fluid {
namespace algorithm {

class DataSetQuery {

public:
  using string = std::string;

  using DataSet = FluidDataSet<string, double, 1>;

  struct Condition{
    index column;
    index comparison;
    double value;

    bool test(RealVectorView point){
      using namespace std;
      switch(comparison){
        case 0: return point(column) == value;
        case 1: return point(column) != value;
        case 2: return point(column) < value;
        case 3: return point(column) <= value;
        case 4: return point(column) > value;
        case 5: return point(column) >= value;
      }
    }
  };

  DataSetQuery()
  {
    mComparisons = std::set<std::string>{"==", "!=", "<", "<=", ">", ">="};
  }

  void addColumn(index col)
  {
    mColumns.emplace(col);
  }

  void addRange(index from, index to)
  {
    for(index i = from; i < to; i++){
      mColumns.emplace(i);
    }
  }

  index numColumns()
  {
    return mColumns.size();
  }

  bool hasAndConditions()
  {
    return (mAndConditions.size() > 0);
  }

  index maxColumn()
  {
    return mColumns.empty()? 0: *mColumns.rbegin();
  }

  bool addCondition(index column, string comparison, double value, bool conjunction)
  {
    auto pos = mComparisons.find(comparison);
    if(pos == mComparisons.end()) return false;
    index i = std::distance(mComparisons.begin(), pos);
    if(conjunction) mAndConditions.emplace_back(Condition{column, i, value});
    else mOrConditions.emplace_back(Condition{column, i, value});
    return true;
  }

  void process (const DataSet& input, DataSet& output){
    auto data = input.getData();
    auto ids = input.getIds();
    for(index i = 0; i < input.size(); i++){
       bool matchesAllAnd = true;
       auto point = data.row(i);
       for(index j = 0; j < mAndConditions.size(); j++)
         if(!mAndConditions[j].test(point)) matchesAllAnd = false;
       if (matchesAllAnd) {
         addRow(ids(i), point, output);
         continue;
       }
       else{
         bool matchesAnyOr = false;
         for(index k = 0; k < mOrConditions.size(); k++)
         if(mOrConditions[k].test(point)) matchesAnyOr = true;
         if (matchesAnyOr) {
           addRow(ids(i), point, output);
         }
     }
   }
 }

  void print() const { }

  void reset() {
    mColumns.clear();
    mAndConditions.clear();
    mOrConditions.clear();
   }

private:

  void addRow(string id, RealVectorView point, DataSet& out){
    RealVector newPoint(mColumns.size());
    newPoint.fill(0);
    index i = 0;
    for(auto c: mColumns) newPoint(i++) = point(c);
    out.add(id, newPoint);
  }

  std::set<index> mColumns;
  std::set<std::string> mComparisons;
  std::vector<Condition> mAndConditions;
  std::vector<Condition> mOrConditions;
};
} // namespace algorithm
} // namespace fluid
