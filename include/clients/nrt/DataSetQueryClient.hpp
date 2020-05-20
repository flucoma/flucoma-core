#pragma once
#include "NRTClient.hpp"
#include "algorithms/DataSetQuery.hpp"

namespace fluid {
namespace client {

class DataSetQueryClient : public FluidBaseClient, OfflineIn, OfflineOut {
  enum { kName };

public:
  using string = std::string;
  using DataSet = FluidDataSet<string, double, 1>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS();

  DataSetQueryClient(ParamSetViewType &p) : mParams(p) {}

  MessageResult<void> addColumn(index column) {
    if (column < 0) return Error("invalid index");
    mAlgorithm.addColumn(column);
    return OK();
  }
  MessageResult<void> addRange(index from, index count) {
    if (from < 0 || count  <= 0) return Error("invalid range");
    mAlgorithm.addRange(from, count);
    return OK();
  }

  MessageResult<void> filter(index column, string comparison, double value) {
    if (column < 0) return Error("invalid index");
    if(mAlgorithm.hasAndConditions()) return Error("Filter already set");
    bool result = mAlgorithm.addCondition(column, comparison, value ,true);
    if(!result) return Error("invalid filter");
    else return OK();
  }

  MessageResult<void> andFilter(index column, string comparison, double value) {
    if (column < 0) return Error("invalid index");
    if(!mAlgorithm.hasAndConditions()) return  Error("Add a filter first");
    bool result = mAlgorithm.addCondition(column, comparison, value, true);
    if(!result) return Error("invalid filter");
    else return OK();
  }

  MessageResult<void> orFilter(index column, string comparison, double value) {
    if (column < 0) return Error("invalid index");
    if(!mAlgorithm.hasAndConditions()) return Error("Add a filter first");
    bool result = mAlgorithm.addCondition(column, comparison, value, false);
    if(!result) return Error("invalid filter");
    else return OK();
  }


  MessageResult<void> transform(DataSetClientRef sourceClient, DataSetClientRef destClient) {
    if(mAlgorithm.numColumns() <= 0) return Error("No columns");
    auto srcPtr = sourceClient.get().lock();
    auto destPtr = destClient.get().lock();
    if(!srcPtr || !destPtr) return Error(NoDataSet);
    auto src = srcPtr->getDataSet();
    if (src.size() == 0) return Error(EmptyDataSet);
    if(src.pointSize() < mAlgorithm.maxColumn()) return Error(WrongPointSize);
    DataSet result(mAlgorithm.numColumns());
    mAlgorithm.process(src, result);
    destPtr->setDataSet(result);
    return OK();
  }

  MessageResult<void> reset() {
    mAlgorithm.reset();
    return OK();
  }

  MessageResult<void> limit(index rows) {
    if (rows <= 0) return Error("invalid value");
    mAlgorithm.limit(rows);
    return OK();
  }

  FLUID_DECLARE_MESSAGES(
                        makeMessage("transform", &DataSetQueryClient::transform),
                         makeMessage("addColumn", &DataSetQueryClient::addColumn),
                         makeMessage("addRange", &DataSetQueryClient::addRange),
                         makeMessage("filter", &DataSetQueryClient::filter),
                         makeMessage("and", &DataSetQueryClient::andFilter),
                         makeMessage("or", &DataSetQueryClient::orFilter),
                         makeMessage("reset", &DataSetQueryClient::reset),
                         makeMessage("limit", &DataSetQueryClient::limit)
  );

private:
  algorithm::DataSetQuery mAlgorithm;
};
using NRTThreadedDataSetQueryClient = NRTThreadingAdaptor<ClientWrapper<DataSetQueryClient>>;
} // namespace client
} // namespace fluid
