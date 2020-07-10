#pragma once

#include "DataSetClient.hpp"
#include "NRTClient.hpp"
#include "algorithms/Standardization.hpp"

namespace fluid {
namespace client {

class StandardizeClient : public FluidBaseClient,
                          AudioIn,
                          ControlOut,
                          ModelObject,
                          public DataClient<algorithm::Standardization> {

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using StringVector = FluidTensor<string, 1>;

  template <typename T> Result process(FluidContext &) { return {}; }
  enum { kInputBuffer, kOutputBuffer };
  FLUID_DECLARE_PARAMS(BufferParam("inputPointBuffer", "Input Point Buffer"),
                       BufferParam("predictionBuffer", "Prediction Buffer"));

  StandardizeClient(ParamSetViewType &p) : mParams(p) {
    audioChannelsIn(1);
    controlChannelsOut(1);
  }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>> &input,
               std::vector<FluidTensorView<T, 1>> &output, FluidContext &) {
    if (!mAlgorithm.initialized()) return;
    InOutBuffersCheck bufCheck(mAlgorithm.dims());
    if (!bufCheck.checkInputs(get<kInputBuffer>().get(), get<kOutputBuffer>().get()))
      return;
    RealVector src(mDims);
    RealVector dest(mDims);
    src = BufferAdaptor::ReadAccess(get<kInputBuffer>().get()).samps(0, mDims, 0);
    mTrigger.process(input, output, [&]() {
      mAlgorithm.processFrame(src, dest);
      BufferAdaptor::Access(get<kOutputBuffer>().get()).samps(0) = dest;
    });
  }

  index latency() { return 0; }

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
    if (!mAlgorithm.initialized()) return Error(NoDataFitted);
    InOutBuffersCheck bufCheck(mAlgorithm.dims());
    if (!bufCheck.checkInputs(in.get(), out.get())) return Error(bufCheck.error());
    BufferAdaptor::Access outBuf(out.get());
    Result resizeResult = outBuf.resize(mDims, 1, outBuf.sampleRate());
    if (!resizeResult.ok())
      return Error(BufferAlloc);
    RealVector src(mDims);
    RealVector dest(mDims);
    src = BufferAdaptor::ReadAccess(in.get()).samps(0, mDims, 0);
    mAlgorithm.processFrame(src, dest);
    outBuf.samps(0) = dest;
    return OK();
  }

  MessageResult<void> fitTransform(DataSetClientRef sourceClient,
                                   DataSetClientRef destClient) {
    auto result = fit(sourceClient);
    if (!result.ok())
      return result;
    result = transform(sourceClient, destClient);
    return result;
  }

  FLUID_DECLARE_MESSAGES(
      makeMessage("fit", &StandardizeClient::fit),
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
  index mDims{0};
  FluidInputTrigger mTrigger;
};

using RTStandardizeClient = ClientWrapper<StandardizeClient>;

} // namespace client
} // namespace fluid
