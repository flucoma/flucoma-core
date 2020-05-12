#pragma once
#include "CommonResults.hpp"
#include "FluidSharedInstanceAdaptor.hpp"
#include "clients/common/SharedClientUtils.hpp"
#include "data/FluidDataSet.hpp"
#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/FluidNRTClientWrapper.hpp>
#include <clients/common/MessageSet.hpp>
#include <clients/common/OfflineClient.hpp>
#include <clients/common/ParameterSet.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/Result.hpp>
#include <data/FluidFile.hpp>
#include <data/FluidTensor.hpp>
#include <data/FluidIndex.hpp>
#include <data/TensorTypes.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

namespace fluid {
namespace client {

class DataSetClient : public FluidBaseClient, OfflineIn, OfflineOut {
  enum { kName };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using DataSet = FluidDataSet<string, double, 1>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS(StringParam<Fixed<true>>("name", "DataSet"));

  DataSetClient(ParamSetViewType &p) : mParams(p), mDataSet(0) {}

  MessageResult<void> addPoint(string id, BufferPtr data) {
    if (!data)
      return NoBufferError;
    BufferAdaptor::Access buf(data.get());

    if (mDataSet.size() == 0) {
      if (mDataSet.pointSize() != buf.numFrames())
        mDataSet = DataSet(buf.numFrames());
      mDims = mDataSet.pointSize();
    } else if (buf.numFrames() != mDims)
      return WrongPointSizeError;

    FluidTensor<double, 1> point(mDims);
    point = buf.samps(0, mDims, 0);
    return mDataSet.add(id, point) ? OKResult : DuplicateError;
  }

  MessageResult<void> getPoint(string id, BufferPtr data) const {
    if (!data)
      return NoBufferError;
    BufferAdaptor::Access buf(data.get());
    Result resizeResult = buf.resize(mDims, 1, buf.sampleRate());
    if (!resizeResult.ok())
      return {resizeResult.status(), resizeResult.message()};
    FluidTensor<double, 1> point(mDims);
    point = buf.samps(0, mDims, 0);
    bool result = mDataSet.get(id, point);
    //    mDataSet.print();
    if (result) {
      buf.samps(0, mDims, 0) = point;
      return OKResult;
    } else {
      return PointNotFoundError;
    }
  }

  MessageResult<void> updatePoint(string id, BufferPtr data) {
    if (!data)
      return NoBufferError;
    BufferAdaptor::Access buf(data.get());
    if (buf.numFrames() < mDims)
      return WrongPointSizeError;
    FluidTensor<double, 1> point(mDims);
    point = buf.samps(0, mDims, 0);
    return mDataSet.update(id, point) ? OKResult : PointNotFoundError;
  }

  MessageResult<void> deletePoint(string id) {
    return mDataSet.remove(id) ? OKResult : PointNotFoundError;
  }

  MessageResult<index> size() { return mDataSet.size(); }
  MessageResult<index> cols() { return mDataSet.pointSize(); }

  MessageResult<void> clear() {
    mDataSet = DataSet(0);
    return OKResult;
  }

  MessageResult<void> write(string fileName) {
    auto file = FluidFile(fileName, "w");
    if (!file.valid()) {
      return {Result::Status::kError, file.error()};
    }
    file.add("ids", mDataSet.getIds());
    file.add("data", mDataSet.getData());
    file.add("cols", mDataSet.pointSize());
    file.add("rows", mDataSet.size());
    return file.write() ? OKResult : WriteError;
  }

  MessageResult<void> read(string fileName) {
    auto file = FluidFile(fileName, "r");
    if (!file.valid()) {
      return {Result::Status::kError, file.error()};
    }
    if (!file.read()) {
      return ReadError;
    }
    if (!file.checkKeys({"data", "ids", "rows", "cols"})) {
      return {Result::Status::kError, file.error()};
    }
    index cols, rows;
    file.get("cols", cols);
    file.get("rows", rows);
    FluidTensor<string, 1> ids(rows);
    FluidTensor<double, 2> data(rows, cols);
    file.get("ids", ids, rows);
    file.get("data", data, rows, cols);
    mDataSet = DataSet(ids, data);
    mDims = cols;
    return OKResult;
  }

  FLUID_DECLARE_MESSAGES(
      makeMessage("addPoint", &DataSetClient::addPoint),
      makeMessage("getPoint", &DataSetClient::getPoint),
      makeMessage("updatePoint", &DataSetClient::updatePoint),
      makeMessage("deletePoint", &DataSetClient::deletePoint),
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
  index mDims;
};
using DataSetClientRef = SharedClientRef<DataSetClient>;
using NRTThreadedDataSetClient =
    NRTThreadingAdaptor<typename DataSetClientRef::SharedType>;

} // namespace client
} // namespace fluid
