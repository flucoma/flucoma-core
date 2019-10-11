#pragma once
#include "DatasetErrorStrings.hpp"
#include "FluidSharedInstanceAdaptor.hpp"
#include "clients/common/SharedClientUtils.hpp"
#include "data/FluidDataset.hpp"
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

class LabelsetClient : public FluidBaseClient, OfflineIn, OfflineOut {
  enum { kName };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using LabelledDataset = FluidDataset<string, double, string, 1>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS(StringParam<Fixed<true>>("name", "Dataset"));

  LabelsetClient(ParamSetViewType &p) : mParams(p), mDataset(1) {
    mDims = 1;
  }


  // TODO: refactor with addPoint
  MessageResult<void> addLabel(string id, string label) {
    if(id.empty()) return {Result::Status::kError, "Empty id"};
    if(label.empty()) return {Result::Status::kError, "Empty label"};
    FluidTensor<double, 1> point(mDims);
    return mDataset.add(id, point, label) ? mOKResult : mDuplicateError;
  }

  MessageResult<string> getLabel(string id) const {
    if(id.empty()) return {Result::Status::kError, "Empty id"};
    FluidTensor<double, 1> point(mDims);
    return  mDataset.getTarget(id);
  }

  /*MessageResult<void> updateLabel(string id, string label) {
    if(id.empty()) {Result::Status::kError, "Empty id"};
    //if(label.empty()) {Result::Status::kError, "Empty label"};

    return mDataset.update(label, point) ? mOKResult : mNotFoundError;
  }*/

  MessageResult<void> deleteLabel(string id) {
    return mDataset.remove(id) ? mOKResult : mNotFoundError;
  }

  MessageResult<int> size() { return mDataset.size(); }

  MessageResult<void> clear() {
    mDataset = LabelledDataset(mDims);
    return mOKResult;
  }

  MessageResult<void> write(string fileName) {
    auto file = FluidFile(fileName, "w");
    if(!file.valid()){return {Result::Status::kError, file.error()};}
    file.add("labels", mDataset.getTargets());
    file.add("ids", mDataset.getIds());
    file.add("rows", mDataset.size());
    /*file.add("data", mDataset.getData());
    file.add("cols", mDataset.pointSize());
    file.add("rows", mDataset.size());*/
    return file.write()? mOKResult:mWriteError;
  }

 MessageResult<void> read(string fileName) {
   auto file = FluidFile(fileName, "r");
   if(!file.valid()){return {Result::Status::kError, file.error()};}
   if(!file.read()){return {Result::Status::kError, ReadError};}
   if(!file.checkKeys({"targets","ids","rows"})){
     return {Result::Status::kError, file.error()};
   }
   size_t  rows;
   file.get("rows", rows);
   FluidTensor<string, 1> ids(rows);
   FluidTensor<string, 1> targets(rows);
   FluidTensor<double, 2> data(rows,1);
   file.get("ids", ids, rows);
   file.get("targets", targets, rows);
   mDataset = LabelledDataset(ids, data, targets);
   return mOKResult;
 }

  FLUID_DECLARE_MESSAGES(
      makeMessage("addLabel", &LabelsetClient::addLabel),
      makeMessage("getLabel", &LabelsetClient::getLabel),
      makeMessage("deleteLabel", &LabelsetClient::deleteLabel),
      makeMessage("size", &LabelsetClient::size),
      makeMessage("clear", &LabelsetClient::clear),
      makeMessage("write", &LabelsetClient::write),
      makeMessage("read", &LabelsetClient::read)
  );

  const LabelledDataset getDataset() const { return mDataset; }

private:
  using result = MessageResult<void>;
  //result mNoBufferError{Result::Status::kError, NoBufferError};
  result mWriteError{Result::Status::kError, WriteError};
  result mNotFoundError{Result::Status::kError, PointNotFoundError};
  //result mWrongSizeError{Result::Status::kError, WrongPointSizeError};
  result mDuplicateError{Result::Status::kError,DuplicateError};
  result mOKResult{Result::Status::kOk};

  mutable LabelledDataset mDataset;
  size_t mDims;
};
using LabelsetClientRef = SharedClientRef<LabelsetClient>;
using NRTThreadedLabelsetClient =
    NRTThreadingAdaptor<typename LabelsetClientRef::SharedType>;

} // namespace client
} // namespace fluid
