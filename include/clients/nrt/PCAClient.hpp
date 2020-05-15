#pragma once
#include "NRTClient.hpp"
#include "algorithms/PCA.hpp"

namespace fluid {
namespace client {

class PCAClient : public FluidBaseClient, OfflineIn, OfflineOut {

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using StringVector = FluidTensor<string, 1>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS();

  PCAClient(ParamSetViewType &p) : mParams(p) {}

  MessageResult<void> fit(DataSetClientRef datasetClient, index k) {
    auto datasetClientPtr = datasetClient.get().lock();
    if(!datasetClientPtr) return Error(NoDataSet);
    auto dataSet = datasetClientPtr->getDataSet();
    if (dataSet.size() == 0) return Error(EmptyDataSet);
    if (k <= 0) return Error(SmallK);

    mDims = dataSet.pointSize();
    mK = k;
    mAlgorithm.init(dataSet.getData(), k);
    return OK();
  }

  MessageResult<index> cols() { return mK; }
  MessageResult<index> rows() { return mDims; }

  MessageResult<void> fitTransform(DataSetClientRef sourceClient,
                                   DataSetClientRef destClient, index k) {
            auto result = fit(sourceClient, k);
            if (!result.ok()) return result;
            result = transform(sourceClient, destClient);
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
        return Error(EmptyDataSet);
      StringVector ids{srcDataSet.getIds()};
      RealMatrix output(srcDataSet.size(), mK);
      if (!mAlgorithm.initialized())
        return Error(NoDataFitted);
      mAlgorithm.process(srcDataSet.getData(), output);
      FluidDataSet<string, double, 1> result(ids, output);
      destPtr->setDataSet(result);
    } else {
      return Error(NoDataSet);
    }
    return {};
  }

  MessageResult<void> transformPoint(BufferPtr in, BufferPtr out) const {
    if (!in || !out)
      return Error(NoBuffer);
    BufferAdaptor::Access inBuf(in.get());
    BufferAdaptor::Access outBuf(out.get());
    if (inBuf.numFrames() != mDims)
      return Error(WrongPointSize);
    if (!mAlgorithm.initialized())
      return Error(NoDataFitted);
    Result resizeResult = outBuf.resize(mK, 1, inBuf.sampleRate());
    if (!resizeResult.ok())
      return Error(BufferAlloc);
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
      return Error(file.error());
    }
    RealMatrix bases(mDims, mK);
    mAlgorithm.getBases(bases);
    RealVector mean(mDims);
    mAlgorithm.getMean(mean);
    file.add("bases", bases);
    file.add("mean", mean);
    file.add("rows", mDims);
    file.add("cols", mK);
    return file.write() ? OK() : Error(FileWrite);
  }

  MessageResult<void> read(string fileName) {
    auto file = FluidFile(fileName, "r");
    if (!file.valid()) {
      return Error(file.error());
    }
    if (!file.read()) {
      return Error(FileRead);
    }
    if (!file.checkKeys({"bases", "mean", "cols", "rows"})) {
      return Error(file.error());
    }
    file.get("rows", mDims);
    file.get("cols", mK);
    RealMatrix bases(mDims, mK);
    file.get("bases", bases, mDims, mK);
    RealVector mean(mDims);
    file.get("mean", mean, mDims);
    mAlgorithm.init(bases, mean);
    return OK();
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
  algorithm::PCA mAlgorithm;
  index mDims;
  index mK;
};

using NRTThreadedPCAClient = NRTThreadingAdaptor<ClientWrapper<PCAClient>>;

} // namespace client
} // namespace fluid
