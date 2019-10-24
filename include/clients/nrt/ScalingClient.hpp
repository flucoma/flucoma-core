#pragma once

//#include "DataSetClient.hpp"
#include "DataSetErrorStrings.hpp"
#include "algorithms/Scaling.hpp"
#include "data/FluidDataSet.hpp"

#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/MessageSet.hpp>
#include <clients/common/OfflineClient.hpp>
#include <clients/common/ParameterSet.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/Result.hpp>
#include <clients/nrt/FluidNRTClientWrapper.hpp>
#include <data/FluidFile.hpp>
#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>
#include <nlohmann/json.hpp>
#include <string>

namespace fluid {
namespace client {

class ScalingClient : public FluidBaseClient, OfflineIn, OfflineOut {
  enum { kNDims, kMin, kMax };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS(LongParam<Fixed<true>>("nDims", "Dimension size", 1,
                                              Min(1)),
                       FloatParam("min", "Minimum value", 0.0),
                       FloatParam("max", "Maximum value", 1.0, LowerLimit<kMin>()));

  ScalingClient(ParamSetViewType &p)
      : mParams(p), mDims(get<kNDims>()), mDataSet(get<kNDims>()) {
  }

  MessageResult<void> fit(DataSetClientRef datasetClient) {
    auto weakPtr = datasetClient.get();
    if (auto datasetClientPtr = weakPtr.lock()) {
      auto dataset = datasetClientPtr->getDataSet();
      if (dataset.size() == 0)
        return {Result::Status::kError, EmptyDataSetError};
      mAlgorithm.init(get<kMin>(), get<kMax>(), dataset.getData());
    } else {
      return {Result::Status::kError, "DataSet doesn't exist"};
    }
    return {};
  }

  MessageResult<void> scale(DataSetClientRef sourceClient,
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
      if(!mAlgorithm.initialized())return {Result::Status::kError, "No data fitted"};
      mAlgorithm.setMin(get<kMin>());
      mAlgorithm.setMax(get<kMax>());
      mAlgorithm.process(srcDataSet.getData(), data);
      FluidDataSet<string, double, 1> result(ids, data);
      destPtr->setDataSet(result);
    } else {
      return {Result::Status::kError, "DataSet doesn't exist"};
    }
    return {};
  }

  MessageResult<void> scalePoint(BufferPtr in, BufferPtr out) const {
    if (!in || !out)
      return {Result::Status::kError, NoBufferError};
    BufferAdaptor::Access inBuf(in.get());
    BufferAdaptor::Access outBuf(out.get());
    if (inBuf.numFrames() != mDims)
      return {Result::Status::kError, WrongPointSizeError};
    if(!mAlgorithm.initialized())return {Result::Status::kError, "No data fitted"};
    Result resizeResult = outBuf.resize(mDims, 1, inBuf.sampleRate());
    if(!resizeResult.ok()) return {Result::Status::kError, "Cant allocsate buffer"};
    FluidTensor<double, 1> src(mDims);
    FluidTensor<double, 1> dest(mDims);
    src = inBuf.samps(0, mDims, 0);
    mAlgorithm.setMin(get<kMin>());
    mAlgorithm.setMax(get<kMax>());
    mAlgorithm.processFrame(src, dest);
    outBuf.samps(0) = dest;
    return {};
  }

  MessageResult<void> write(string fileName) {
    auto file = FluidFile(fileName, "w");
    if (!file.valid()) {
      return {Result::Status::kError, file.error()};
    }
    RealVector min(mDims);
    RealVector max(mDims);
    mAlgorithm.getDataMin(min);
    mAlgorithm.getDataMax(max);
    file.add("min", min);
    file.add("max", max);
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
    if (!file.checkKeys({"min", "max", "cols"})) {
      return {Result::Status::kError, file.error()};
    }
    RealVector dataMin(mDims);
    RealVector dataMax(mDims);
    size_t dims;
    file.get("cols", dims);
    if (dims!=mDims)return {Result::Status::kError, WrongPointSizeError};
    file.get("min", dataMin, dims);
    file.get("max", dataMax, dims);
    mAlgorithm.init(get<kMin>(), get<kMax>(), dataMin, dataMax);
    return mOKResult;
  }

  FLUID_DECLARE_MESSAGES(makeMessage("fit", &ScalingClient::fit),
                         makeMessage("scale", &ScalingClient::scale),
                         makeMessage("scalePoint", &ScalingClient::scalePoint),
                         makeMessage("read", &ScalingClient::read),
                         makeMessage("write", &ScalingClient::write));

private:
  MessageResult<void> mOKResult{Result::Status::kOk};
  MessageResult<void> mWriteError{Result::Status::kError, WriteError};
  mutable FluidDataSet<string, double, 1> mDataSet;
  mutable algorithm::Scaling mAlgorithm;
  size_t mDims;
};

using NRTThreadedScalingClient =
    NRTThreadingAdaptor<ClientWrapper<ScalingClient>>;

} // namespace client
} // namespace fluid
