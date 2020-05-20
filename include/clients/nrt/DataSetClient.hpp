#pragma once
#include "NRTClient.hpp"
#include "../common/SharedClientUtils.hpp"
#include "data/FluidDataSet.hpp"
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>

namespace fluid {
namespace client {

class DataSetClient : public FluidBaseClient, OfflineIn, OfflineOut {
  enum { kName };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using DataSet = FluidDataSet<string, double, 1>;
  using StringVector = FluidTensor<string, 1>;
  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS(StringParam<Fixed<true>>("name", "DataSet"));

  DataSetClient(ParamSetViewType &p) : mParams(p), mDataSet(0) {}

  MessageResult<void> addPoint(string id, BufferPtr data) {
    if (!data)
      return Error(NoBuffer);
    BufferAdaptor::Access buf(data.get());
    if(!buf.exists()) return Error(InvalidBuffer);
    if (mDataSet.size() == 0) {
      if (mDataSet.pointSize() != buf.numFrames())
        mDataSet = DataSet(buf.numFrames());
      mDims = mDataSet.pointSize();
    } else if (buf.numFrames() != mDims)
      return Error(WrongPointSize);
    RealVector point(mDims);
    point = buf.samps(0, mDims, 0);
    return mDataSet.add(id, point) ? OK() : Error(DuplicateLabel);
  }

  MessageResult<void> getPoint(string id, BufferPtr data) const {
    if (!data)
      return Error(NoBuffer);
    BufferAdaptor::Access buf(data.get());
    if(!buf.exists()) return Error(InvalidBuffer);
    Result resizeResult = buf.resize(mDims, 1, buf.sampleRate());
    if (!resizeResult.ok())
      return {resizeResult.status(), resizeResult.message()};
    RealVector point(mDims);
    point = buf.samps(0, mDims, 0);
    bool result = mDataSet.get(id, point);
    if (result) {
      buf.samps(0, mDims, 0) = point;
      return OK();
    } else {
      return Error(PointNotFound);
    }
  }

  MessageResult<void> updatePoint(string id, BufferPtr data) {
    if (!data)
      return Error(NoBuffer);
    BufferAdaptor::Access buf(data.get());
    if(!buf.exists()) return Error(InvalidBuffer);
    if (buf.numFrames() < mDims)
      return Error(WrongPointSize);
    RealVector point(mDims);
    point = buf.samps(0, mDims, 0);
    return mDataSet.update(id, point) ? OK() : Error(PointNotFound);
  }

  MessageResult<void> deletePoint(string id) {
    return mDataSet.remove(id) ?  OK() : Error(PointNotFound);
  }

  MessageResult<index> size() { return mDataSet.size(); }
  MessageResult<index> cols() { return mDataSet.pointSize(); }

  MessageResult<void> clear() {
    mDataSet = DataSet(0);
    return {};
  }

  MessageResult<void> write(string fileName) {
    auto file = FluidFile(fileName, "w");
    if (!file.valid()) {
      return Error(file.error());
    }
    file.add("ids", mDataSet.getIds());
    file.add("data", mDataSet.getData());
    file.add("cols", mDataSet.pointSize());
    file.add("rows", mDataSet.size());
    return file.write() ? OK() : Error(FileWrite);
  }

  MessageResult<void> read(string fileName) {
    auto file = FluidFile(fileName, "r");
    if (!file.valid()) return Error(file.error());
    if (!file.read()) return Error(FileRead);

    if (!file.checkKeys({"data", "ids", "rows", "cols"})) {
      return Error(file.error());
    }
    index cols, rows;
    file.get("cols", cols);
    file.get("rows", rows);
    StringVector ids(rows);
    RealMatrix data(rows, cols);
    file.get("ids", ids, rows);
    file.get("data", data, rows, cols);
    mDataSet = DataSet(ids, data);
    mDims = cols;
    return OK();
  }

    string  printRow(RealVectorView row, index maxCols){
      using namespace std;
      ostringstream result;
      if(row.size() < maxCols) {
        for(index c = 0; c < row.size();c++){
          result << setw(10) << setprecision(5) << row(c);
        }
      }
      else{
        for(index c = 0; c < maxCols / 2; c++){
          result << setw(10) << setprecision(5) << row(c);
        }
        result << setw(10) << "..";
        for(index c = maxCols / 2; c > 0; c--){
          result << setw(10) << setprecision(5) << row(row.size() - c);
        }
      }
      return result.str();
    }
    MessageResult<string> print() {
      using namespace std;
      if (mDataSet.size() == 0) return {"Empty dataset"};
      auto ids = mDataSet.getIds();
      auto data = mDataSet.getData();
      ostringstream result;
      result << std::endl  <<
        "rows: "<< mDataSet.size() <<
        " cols: "<< mDataSet.pointSize() << std::endl;
      index maxRows = 6, maxCols = 6;
      if(mDataSet.size() < maxRows) {
        for(index r = 0; r < mDataSet.size();r++){
          result << ids(r) <<" "<<printRow(data.row(r), maxCols)<<std::endl;
        }
      }
      else{
        for(index r = 0; r < maxRows/2;r++){
          result << ids(r) << " " <<
          printRow(data.row(r), maxCols) << std::endl;
        }
        result << setw(10) << "..." << std::endl;;
        for(index r = maxRows/2; r > 0;r--){
          result << ids(mDataSet.size() - r) << " " <<
          printRow(data.row(mDataSet.size() - r), maxCols) << std::endl;
        }
      }
      return result.str();
    }

  MessageResult<string> dump() {
    using json = nlohmann::json;
    using namespace std;
    string result;
    auto rowArray = json::array();
    auto ids = mDataSet.getIds();
    auto data = mDataSet.getData();
    for(index r = 0; r < mDataSet.size();r++){
      auto row = data.row(r);
      auto rowV = vector<double>(row.begin(), row.end());
      auto rowObj = json::object({{"id",ids(r)},{"data",rowV}});
      rowArray.push_back(rowObj);
    }
    json j ;
    j["rows"] = mDataSet.size();
    j["cols"] = mDataSet.pointSize();
    j["data"] = rowArray;
    result = j.dump();
    return result;
  }

  FLUID_DECLARE_MESSAGES(
      makeMessage("addPoint", &DataSetClient::addPoint),
      makeMessage("getPoint", &DataSetClient::getPoint),
      makeMessage("updatePoint", &DataSetClient::updatePoint),
      makeMessage("deletePoint", &DataSetClient::deletePoint),
      makeMessage("dump", &DataSetClient::dump),
      makeMessage("print", &DataSetClient::print),
      makeMessage("size", &DataSetClient::size),
      makeMessage("cols", &DataSetClient::cols),
      makeMessage("clear", &DataSetClient::clear),
      makeMessage("write", &DataSetClient::write),
      makeMessage("read", &DataSetClient::read));

  const DataSet getDataSet() const { return mDataSet; }

  void setDataSet(DataSet ds) {
    mDataSet = ds;
    mDims = ds.pointSize();
  }

private:
  DataSet mDataSet;
  index mDims{0};
};
using DataSetClientRef = SharedClientRef<DataSetClient>;
using NRTThreadedDataSetClient =
    NRTThreadingAdaptor<typename DataSetClientRef::SharedType>;

} // namespace client
} // namespace fluid
