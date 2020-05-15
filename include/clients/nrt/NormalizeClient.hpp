#pragma once

#include "NRTClient.hpp"
#include "algorithms/Normalization.hpp"

namespace fluid {
namespace client {

class NormalizeClient : public FluidBaseClient, OfflineIn, OfflineOut {
  enum {kMin, kMax};

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using StringVector = FluidTensor<string, 1>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS(FloatParam("min", "Minimum value", 0.0),
                       FloatParam("max", "Maximum value", 1.0, LowerLimit<kMin>()));

  NormalizeClient(ParamSetViewType &p)
      : mParams(p) {
  }

  MessageResult<void> fit(DataSetClientRef datasetClient) {
    auto weakPtr = datasetClient.get();
    if (auto datasetClientPtr = weakPtr.lock()) {
      auto dataset = datasetClientPtr->getDataSet();
      if (dataset.size() == 0)
        return Error(EmptyDataSet);
      mDims = dataset.pointSize();
      mAlgorithm.init(get<kMin>(), get<kMax>(), dataset.getData());
    } else {
      return Error(NoDataSet);
    }
    return {};
  }

  MessageResult<index> cols() { return mDims;}

  MessageResult<void> transform(DataSetClientRef sourceClient,
                            DataSetClientRef destClient) {
    using namespace std;
    auto srcPtr = sourceClient.get().lock();
    auto destPtr = destClient.get().lock();
    if (srcPtr && destPtr) {
      auto srcDataSet = srcPtr->getDataSet();
      if (srcDataSet.size() == 0)
        return Error(EmptyDataSet);
      StringVector ids{srcDataSet.getIds()};
      RealMatrix data(srcDataSet.size(), srcDataSet.pointSize());
      if(!mAlgorithm.initialized()) return Error(NoDataFitted);
      mAlgorithm.setMin(get<kMin>());
      mAlgorithm.setMax(get<kMax>());
      mAlgorithm.process(srcDataSet.getData(), data);
      FluidDataSet<string, double, 1> result(ids, data);
      destPtr->setDataSet(result);
    } else {
      return Error(NoDataSet);
    }
    return {};
  }

  MessageResult<void> fitTransform(DataSetClientRef sourceClient,
                                   DataSetClientRef destClient) {
            auto result = fit(sourceClient);
            if (!result.ok()) return result;
            result = transform(sourceClient, destClient);
            return result;
  }

  MessageResult<void> transformPoint(BufferPtr in, BufferPtr out) {
    if (!in || !out) return Error(NoBuffer);
    BufferAdaptor::Access inBuf(in.get());
    BufferAdaptor::Access outBuf(out.get());
    if (inBuf.numFrames() != mDims) return Error(WrongPointSize);
    if(!mAlgorithm.initialized()) return Error(NoDataFitted);
    Result resizeResult = outBuf.resize(mDims, 1, inBuf.sampleRate());
    if(!resizeResult.ok()) return Error(BufferAlloc);
    RealVector src(mDims);
    RealVector dest(mDims);
    src = inBuf.samps(0, mDims, 0);
    mAlgorithm.setMin(get<kMin>());
    mAlgorithm.setMax(get<kMax>());
    mAlgorithm.processFrame(src, dest);
    outBuf.samps(0) = dest;
    return {};
  }

  MessageResult<void> write(string fileName) {
    auto file = FluidFile(fileName, "w");
    if (!file.valid()) return Error(file.error());
    RealVector min(mDims);
    RealVector max(mDims);
    mAlgorithm.getDataMin(min);
    mAlgorithm.getDataMax(max);
    file.add("min", min);
    file.add("max", max);
    file.add("cols", mDims);
    return file.write() ? OK() : Error(FileWrite);
  }

  MessageResult<void> read(string fileName) {
    auto file = FluidFile(fileName, "r");
    if (!file.valid()) return Error(file.error());
    if (!file.read()) return Error(FileRead);
    if (!file.checkKeys({"min", "max", "cols"})) {
      return Error(file.error());
    }
    RealVector dataMin(mDims);
    RealVector dataMax(mDims);
    index dims;
    file.get("cols", dims);
    if (dims!=mDims)return Error(WrongPointSize);
    file.get("min", dataMin, dims);
    file.get("max", dataMax, dims);
    mAlgorithm.init(get<kMin>(), get<kMax>(), dataMin, dataMax);
    return OK();
  }

  FLUID_DECLARE_MESSAGES(makeMessage("fit", &NormalizeClient::fit),
                         makeMessage("cols", &NormalizeClient::cols),
                         makeMessage("fitTransform", &NormalizeClient::fitTransform),
                         makeMessage("transform", &NormalizeClient::transform),
                         makeMessage("transformPoint", &NormalizeClient::transformPoint),
                         makeMessage("read", &NormalizeClient::read),
                         makeMessage("write", &NormalizeClient::write));

private:
  algorithm::Normalization mAlgorithm;
  index mDims{0};
};

using NRTThreadedNormalizeClient =
    NRTThreadingAdaptor<ClientWrapper<NormalizeClient>>;

} // namespace client
} // namespace fluid
