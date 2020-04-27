#pragma once

#include "DataSetClient.hpp"
#include "DataSetErrorStrings.hpp"
#include "algorithms/PCA.hpp"
#include "data/FluidDataSet.hpp"

#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/FluidNRTClientWrapper.hpp>
#include <clients/common/MessageSet.hpp>
#include <clients/common/OfflineClient.hpp>
#include <clients/common/ParameterSet.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/Result.hpp>
#include <data/FluidFile.hpp>
#include <data/FluidIndex.hpp>
#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>
#include <nlohmann/json.hpp>
#include <string>

namespace fluid {
namespace client {

class PCAClient : public FluidBaseClient, OfflineIn, OfflineOut {

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS();

  PCAClient(ParamSetViewType &p) : mParams(p) {}

  MessageResult<void> fit(DataSetClientRef datasetClient, index k) {
    auto weakPtr = datasetClient.get();
    if (auto datasetClientPtr = weakPtr.lock()) {
      auto dataset = datasetClientPtr->getDataSet();
      if (k <= 0)
        return {Result::Status::kError, "k should be at least 1"};
      if (dataset.size() == 0)
        return {Result::Status::kError, EmptyDataSetError};
      mDims = dataset.pointSize();
      mK = k;
      mAlgorithm.init(dataset.getData(), k);
    } else {
      return {Result::Status::kError, "DataSet doesn't exist"};
    }
    return {};
  }

  MessageResult<index> cols() { return mK; }
  MessageResult<index> rows() { return mDims; }

  MessageResult<void> fitTransform(DataSetClientRef datasetClient, index k,
                                   DataSetClientRef destClient) {
            auto result = fit(datasetClient, k);
            if (!result.ok()) return result;
            result = transform(datasetClient, destClient);
            return result;
}

  MessageResult<void> transform(DataSetClientRef sourceClient,
                                DataSetClientRef destClient) const {
    using namespace std;
    auto srcPtr = sourceClient.get().lock();
    auto destPtr = destClient.get().lock();
    if (srcPtr && destPtr) {
      auto srcDataSet = srcPtr->getDataSet();
      if (srcDataSet.size() == 0)
        return {Result::Status::kError, EmptyDataSetError};
      FluidTensor<string, 1> ids{srcDataSet.getIds()};
      FluidTensor<double, 2> output(srcDataSet.size(), mK);
      if (!mAlgorithm.initialized())
        return {Result::Status::kError, "No data fitted"};
      mAlgorithm.process(srcDataSet.getData(), output);
      FluidDataSet<string, double, 1> result(ids, output);
      destPtr->setDataSet(result);
    } else {
      return {Result::Status::kError, "DataSet doesn't exist"};
    }
    return {};
  }

  MessageResult<void> transformPoint(BufferPtr in, BufferPtr out) const {
    if (!in || !out)
      return {Result::Status::kError, NoBufferError};
    BufferAdaptor::Access inBuf(in.get());
    BufferAdaptor::Access outBuf(out.get());
    if (inBuf.numFrames() != mDims)
      return {Result::Status::kError, WrongPointSizeError};
    if (!mAlgorithm.initialized())
      return {Result::Status::kError, "No data fitted"};
    Result resizeResult = outBuf.resize(mK, 1, inBuf.sampleRate());
    if (!resizeResult.ok())
      return {Result::Status::kError, "Cant allocate buffer"};
    FluidTensor<double, 1> src(mDims);
    FluidTensor<double, 1> dest(mK);
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
    RealMatrix bases(mDims, mK);
    mAlgorithm.getBases(bases);
    RealVector mean(mDims);
    mAlgorithm.getMean(mean);
    file.add("bases", bases);
    file.add("mean", mean);
    file.add("rows", mDims);
    file.add("cols", mK);
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
    if (!file.checkKeys({"bases", "mean", "cols", "rows"})) {
      return {Result::Status::kError, file.error()};
    }
    file.get("rows", mDims);
    file.get("cols", mK);
    RealMatrix bases(mDims, mK);
    file.get("bases", bases, mDims, mK);
    RealVector mean(mDims);
    file.get("mean", mean, mDims);
    mAlgorithm.init(bases, mean);
    return mOKResult;
  }

  FLUID_DECLARE_MESSAGES(makeMessage("fit", &PCAClient::fit),
                         makeMessage("transform", &PCAClient::transform),
                         makeMessage("fitTransform", &PCAClient::fitTransform),
                         makeMessage("transformPoint",
                                     &PCAClient::transformPoint),
                         makeMessage("cols", &PCAClient::cols),
                         makeMessage("rows", &PCAClient::rows),
                         makeMessage("read", &PCAClient::read),
                         makeMessage("write", &PCAClient::write));

private:
  MessageResult<void> mOKResult{Result::Status::kOk};
  MessageResult<void> mWriteError{Result::Status::kError, WriteError};
  algorithm::PCA mAlgorithm;
  index mDims;
  index mK;
};

using NRTThreadedPCAClient = NRTThreadingAdaptor<ClientWrapper<PCAClient>>;

} // namespace client
} // namespace fluid
