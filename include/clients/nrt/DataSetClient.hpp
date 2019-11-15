#pragma once
#include "DataSetErrorStrings.hpp"
#include "FluidSharedInstanceAdaptor.hpp"
#include "clients/common/SharedClientUtils.hpp"
#include "data/FluidDataSet.hpp"
#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/MessageSet.hpp>
#include <clients/common/OfflineClient.hpp>
#include <clients/common/ParameterSet.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/Result.hpp>
#include <clients/nrt/FluidNRTClientWrapper.hpp>
#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>
#include <data/FluidFile.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

namespace fluid {
namespace client {

class DataSetClient : public FluidBaseClient, OfflineIn, OfflineOut {
  enum { kName, kNDims };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using DataSet = FluidDataSet<string, double, 1>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS(StringParam<Fixed<true>>("name", "DataSet"),
                       LongParam<Fixed<true>>("nDims", "Dimension size", 1,
                                              Min(1)));

  DataSetClient(ParamSetViewType &p) : mParams(p), mDataSet(get<kNDims>()) {
    mDims = get<kNDims>();
  }

  MessageResult<void> addPoint(string id, BufferPtr data) {
    if (!data)
      return mNoBufferError;
    BufferAdaptor::Access buf(data.get());
    if (buf.numFrames() != mDims)
      return mWrongSizeError;
    FluidTensor<double, 1> point(mDims);
    point = buf.samps(0, mDims, 0);
    return mDataSet.add(id, point) ? mOKResult : mDuplicateError;
  }

  MessageResult<void> getPoint(string id, BufferPtr data) const {
    if (!data)
      return mNoBufferError;
    BufferAdaptor::Access buf(data.get());
    if (buf.numFrames() < mDims)
      return mWrongSizeError;
    FluidTensor<double, 1> point(mDims);
    point = buf.samps(0, mDims, 0);
    bool result = mDataSet.get(id, point);
    mDataSet.print();
    if (result) {
      buf.samps(0, mDims, 0) = point;
      return {Result::Status::kOk};
    } else {
      return mNotFoundError;
    }
  }

  MessageResult<void> updatePoint(string id, BufferPtr data) {
    if (!data)
      return mNoBufferError;
    BufferAdaptor::Access buf(data.get());
    if (buf.numFrames() < mDims)
      return mWrongSizeError;
    FluidTensor<double, 1> point(mDims);
    point = buf.samps(0, mDims, 0);
    return mDataSet.update(id, point) ? mOKResult : mNotFoundError;
  }

  MessageResult<void> deletePoint(string id) {
    return mDataSet.remove(id) ? mOKResult : mNotFoundError;
  }

  MessageResult<int> size() { return mDataSet.size(); }

  MessageResult<void> clear() {
    mDataSet = DataSet(get<kNDims>());
    return mOKResult;
  }

  MessageResult<void> write(string fileName) {
    auto file = FluidFile(fileName, "w");
    if(!file.valid()){return {Result::Status::kError, file.error()};}
    file.add("ids", mDataSet.getIds());
    file.add("data", mDataSet.getData());
    file.add("cols", mDataSet.pointSize());
    file.add("rows", mDataSet.size());
    return file.write()? mOKResult:mWriteError;
  }

 MessageResult<void> read(string fileName) {
   auto file = FluidFile(fileName, "r");
   if(!file.valid()){return {Result::Status::kError, file.error()};}
   if(!file.read()){return {Result::Status::kError, ReadError};}
   if(!file.checkKeys({"data","ids","rows","cols"})){
     return {Result::Status::kError, file.error()};
   }
   size_t cols, rows;
   file.get("cols", cols);
   file.get("rows", rows);
   FluidTensor<string, 1> ids(rows);
   FluidTensor<double, 2> data(rows,cols);
   file.get("ids", ids, rows);
   file.get("data", data, rows, cols);
   mDataSet = DataSet(ids, data);
   return mOKResult;
 }

  FLUID_DECLARE_MESSAGES(
      makeMessage("addPoint", &DataSetClient::addPoint),
      /*makeMessage("addPointLabel", &DataSetClient::addPointLabel),*/
      makeMessage("getPoint", &DataSetClient::getPoint),
      makeMessage("updatePoint", &DataSetClient::updatePoint),
      makeMessage("deletePoint", &DataSetClient::deletePoint),
      makeMessage("size", &DataSetClient::size),
      makeMessage("clear", &DataSetClient::clear),
      makeMessage("write", &DataSetClient::write),
      makeMessage("read", &DataSetClient::read)
  );

  const DataSet getDataSet() const { return mDataSet; }
  void  setDataSet(DataSet ds) const {mDataSet = ds; }

private:
  using result = MessageResult<void>;
  result mNoBufferError{Result::Status::kError, NoBufferError};
  result mWriteError{Result::Status::kError, WriteError};
  result mNotFoundError{Result::Status::kError, PointNotFoundError};
  result mWrongSizeError{Result::Status::kError, WrongPointSizeError};
  result mDuplicateError{Result::Status::kError,DuplicateError};
  result mOKResult{Result::Status::kOk};

  mutable DataSet mDataSet;
  size_t mDims;
};
using DataSetClientRef = SharedClientRef<DataSetClient>;
using NRTThreadedDataSetClient =
    NRTThreadingAdaptor<typename DataSetClientRef::SharedType>;

} // namespace client
} // namespace fluid
