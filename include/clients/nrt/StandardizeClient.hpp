#pragma once

#include "NRTClient.hpp"
#include "DataSetClient.hpp"
#include "algorithms/Standardization.hpp"

namespace fluid {
namespace client {

class StandardizeClient : public FluidBaseClient, OfflineIn, OfflineOut, ModelObject,
  public DataClient<algorithm::Standardization>  {

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using StringVector = FluidTensor<string, 1>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS();

  StandardizeClient(ParamSetViewType &p)
      : DataClient(mAlgorithm), mParams(p)  {}

  MessageResult<void> fit(DataSetClientRef datasetClient) {
    auto weakPtr = datasetClient.get();
    if (auto datasetClientPtr = weakPtr.lock()) {
      auto dataset = datasetClientPtr->getDataSet();
      if (dataset.size() == 0)
        return Error(EmptyDataSet);
      mDims = dataset.pointSize();
      mAlgorithm.init(dataset.getData());
    } else {
      return Error(NoDataSet);
    }
    return {};
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
      RealMatrix data(srcDataSet.size(), srcDataSet.pointSize());
      if (!mAlgorithm.initialized())
        return Error(NoDataFitted);
      mAlgorithm.process(srcDataSet.getData(), data);
      FluidDataSet<string, double, 1> result(ids, data);
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
    if (inBuf.numFrames() != mDims)
      return Error(WrongPointSize);
    if (!mAlgorithm.initialized())
      return Error(NoDataFitted);
    Result resizeResult = outBuf.resize(mDims, 1, inBuf.sampleRate());
    if (!resizeResult.ok())
      return Error(BufferAlloc);
    RealVector src(mDims);
    RealVector dest(mDims);
    src = inBuf.samps(0, mDims, 0);
    mAlgorithm.processFrame(src, dest);
    outBuf.samps(0) = dest;
    return OK();
  }

  MessageResult<void> fitTransform(DataSetClientRef sourceClient,
                                   DataSetClientRef destClient) {
            auto result = fit(sourceClient);
            if (!result.ok()) return result;
            result = transform(sourceClient, destClient);
            return result;
  }


  FLUID_DECLARE_MESSAGES(makeMessage("fit", &StandardizeClient::fit),
                         makeMessage("fitTransform", &StandardizeClient::fitTransform),
                         makeMessage("transform", &StandardizeClient::transform),
                         makeMessage("transformPoint", &StandardizeClient::transformPoint),
                         makeMessage("cols", &StandardizeClient::dims),
                         makeMessage("size", &StandardizeClient::size),
                         makeMessage("load", &StandardizeClient::load),
                         makeMessage("dump", &StandardizeClient::dump),
                         makeMessage("read", &StandardizeClient::read),
                         makeMessage("write", &StandardizeClient::write));

private:
  algorithm::Standardization mAlgorithm;
  index mDims{0};
};

using NRTThreadedStandardizeClient =
    NRTThreadingAdaptor<ClientWrapper<StandardizeClient>>;

} // namespace client
} // namespace fluid
