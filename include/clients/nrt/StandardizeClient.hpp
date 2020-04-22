#pragma once

#include "DataSetClient.hpp"
#include "DataSetErrorStrings.hpp"
#include "algorithms/Standardization.hpp"
#include "data/FluidDataSet.hpp"

#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/MessageSet.hpp>
#include <clients/common/OfflineClient.hpp>
#include <clients/common/ParameterSet.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/Result.hpp>
#include <clients/common/FluidNRTClientWrapper.hpp>
#include <data/FluidFile.hpp>
#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>
#include <data/FluidIndex.hpp>
#include <nlohmann/json.hpp>
#include <string>

namespace fluid {
namespace client {

class StandardizeClient : public FluidBaseClient, OfflineIn, OfflineOut {
  //enum { };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS();

  StandardizeClient(ParamSetViewType &p)
      : mParams(p) {}

  MessageResult<void> fit(DataSetClientRef datasetClient) {
    auto weakPtr = datasetClient.get();
    if (auto datasetClientPtr = weakPtr.lock()) {
      auto dataset = datasetClientPtr->getDataSet();
      if (dataset.size() == 0)
        return {Result::Status::kError, EmptyDataSetError};
      mDims = dataset.pointSize();
      mAlgorithm.init(dataset.getData());
    } else {
      return {Result::Status::kError, "DataSet doesn't exist"};
    }
    return {};
  }

  MessageResult<index> cols() { return mDims;}

  MessageResult<void> standardize(DataSetClientRef sourceClient,
                            DataSetClientRef destClient) const {
    using namespace std;
    auto srcPtr = sourceClient.get().lock();
    auto destPtr = destClient.get().lock();
    if (srcPtr && destPtr) {
      auto srcDataSet = srcPtr->getDataSet();
      if (srcDataSet.size() == 0)
        return {Result::Status::kError, EmptyDataSetError};
      FluidTensor<string, 1> ids{srcDataSet.getIds()};
      FluidTensor<double, 2> data(srcDataSet.size(), srcDataSet.pointSize());
      if (!mAlgorithm.initialized())
        return {Result::Status::kError, "No data fitted"};
      mAlgorithm.process(srcDataSet.getData(), data);
      FluidDataSet<string, double, 1> result(ids, data);
      destPtr->setDataSet(result);
    } else {
      return {Result::Status::kError, "DataSet doesn't exist"};
    }
    return {};
  }

  MessageResult<void> standardizePoint(BufferPtr in, BufferPtr out) const {
    if (!in || !out)
      return {Result::Status::kError, NoBufferError};
    BufferAdaptor::Access inBuf(in.get());
    BufferAdaptor::Access outBuf(out.get());
    if (inBuf.numFrames() != mDims)
      return {Result::Status::kError, WrongPointSizeError};
    if (!mAlgorithm.initialized())
      return {Result::Status::kError, "No data fitted"};
    Result resizeResult = outBuf.resize(mDims, 1, inBuf.sampleRate());
    if (!resizeResult.ok())
      return {Result::Status::kError, "Cant allocate buffer"};
    FluidTensor<double, 1> src(mDims);
    FluidTensor<double, 1> dest(mDims);
    src = inBuf.samps(0, mDims, 0);
    mAlgorithm.processFrame(src, dest);
    outBuf.samps(0) = dest;
    return {};
  }


  MessageResult<void> write(string fileName) {
    auto file = FluidFile(fileName, "w");
    if (!file.valid()) {
      return {Result::Status::kError, file.error()};
    }
    RealVector mean(mDims);
    RealVector std(mDims);
    mAlgorithm.getMean(mean);
    mAlgorithm.getStd(std);
    file.add("mean", mean);
    file.add("std", std);
    file.add("cols", mDims);
    return file.write() ? mOKResult : mWriteError;
  }

  MessageResult<void> read(string fileName) {
    auto file = FluidFile(fileName, "r");
    if (!file.valid()) {
      return {Result::Status::kError, file.error()};
    }
    if (!file.read()) {
      return {Result::Status::kError, ReadError};
    }
    if (!file.checkKeys({"mean", "std", "cols"})) {
      return {Result::Status::kError, file.error()};
    }
    RealVector mean(mDims);
    RealVector std(mDims);
    index dims;
    file.get("cols", dims);
    if (dims != mDims)
      return {Result::Status::kError, WrongPointSizeError};
    file.get("mean", mean, dims);
    file.get("std", std, dims);
    mAlgorithm.init(mean, std);
    return mOKResult;
  }

  FLUID_DECLARE_MESSAGES(makeMessage("fit", &StandardizeClient::fit),
                         makeMessage("cols", &StandardizeClient::cols),
                         makeMessage("standardize", &StandardizeClient::standardize),
                         makeMessage("standardizePoint", &StandardizeClient::standardizePoint),
                         makeMessage("read", &StandardizeClient::read),
                         makeMessage("write", &StandardizeClient::write));

private:
  MessageResult<void> mOKResult{Result::Status::kOk};
  MessageResult<void> mWriteError{Result::Status::kError, WriteError};
  algorithm::Standardization mAlgorithm;
  index mDims;
};

using NRTThreadedStandardizeClient =
    NRTThreadingAdaptor<ClientWrapper<StandardizeClient>>;

} // namespace client
} // namespace fluid
