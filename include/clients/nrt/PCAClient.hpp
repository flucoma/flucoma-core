#pragma once
#include "NRTClient.hpp"
#include "algorithms/PCA.hpp"
#include "DataSetClient.hpp"


namespace fluid {
namespace client {

class PCAClient : public FluidBaseClient, OfflineIn, OfflineOut, ModelObject,
  public DataClient<algorithm::PCA>  {

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using StringVector = FluidTensor<string, 1>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS();

  PCAClient(ParamSetViewType &p) : mParams(p){}

  MessageResult<void> fit(DataSetClientRef datasetClient, index k) {
    auto datasetClientPtr = datasetClient.get().lock();
    if(!datasetClientPtr) return Error(NoDataSet);
    auto dataSet = datasetClientPtr->getDataSet();
    if (dataSet.size() == 0) return Error(EmptyDataSet);
    if (k <= 0) return Error(SmallK);
    if (dataSet.pointSize() < k) return Error("k is larger than the current dimensions");
    mAlgorithm.init(dataSet.getData(), k);
    return OK();
  }

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
      if (!mAlgorithm.initialized())
        return Error(NoDataFitted);
      StringVector ids{srcDataSet.getIds()};
      RealMatrix output(srcDataSet.size(), mAlgorithm.size());
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
    if(!inBuf.exists()) return Error(InvalidBuffer);
    if(!outBuf.exists()) return Error(InvalidBuffer);
    if (inBuf.numFrames() != mAlgorithm.dims())
      return Error(WrongPointSize);
    if (!mAlgorithm.initialized())
      return Error(NoDataFitted);
    Result resizeResult = outBuf.resize(mAlgorithm.size(), 1, inBuf.sampleRate());
    if (!resizeResult.ok())
      return Error(BufferAlloc);
    FluidTensor<double, 1> src(mAlgorithm.dims());
    FluidTensor<double, 1> dest(mAlgorithm.size());
    src = inBuf.samps(0, mAlgorithm.dims(), 0);
    mAlgorithm.processFrame(src, dest);
    outBuf.samps(0) = dest;
    return {};
  }

  FLUID_DECLARE_MESSAGES(makeMessage("fit", &PCAClient::fit),
                         makeMessage("transform", &PCAClient::transform),
                         makeMessage("fitTransform", &PCAClient::fitTransform),
                         makeMessage("transformPoint",&PCAClient::transformPoint),
                         makeMessage("cols", &PCAClient::dims),
                         makeMessage("size", &PCAClient::size),
                         makeMessage("load", &PCAClient::load),
                         makeMessage("dump", &PCAClient::dump),
                         makeMessage("read", &PCAClient::read),
                         makeMessage("write", &PCAClient::write));
};

using NRTThreadedPCAClient = NRTThreadingAdaptor<ClientWrapper<PCAClient>>;

} // namespace client
} // namespace fluid
