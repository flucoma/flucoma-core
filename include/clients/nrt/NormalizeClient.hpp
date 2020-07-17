#pragma once

#include "DataSetClient.hpp"
#include "NRTClient.hpp"
#include "algorithms/Normalization.hpp"

namespace fluid {
namespace client {

class NormalizeClient : public FluidBaseClient,
                        AudioIn,
                        ControlOut,
                        ModelObject,
                        public DataClient<algorithm::Normalization> {

  enum { kMin, kMax, kInputBuffer, kOutputBuffer };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using StringVector = FluidTensor<string, 1>;

  FLUID_DECLARE_PARAMS(FloatParam("min", "Minimum Value", 0.0),
                       FloatParam("max", "Maximum Value", 1.0),
                       BufferParam("inputPointBuffer", "Input Point Buffer"),
                       BufferParam("predictionBuffer", "Prediction Buffer"));

  NormalizeClient(ParamSetViewType &p) : mParams(p) {
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
    mAlgorithm.setMin(get<kMin>());
    mAlgorithm.setMax(get<kMax>());
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
      mAlgorithm.init(get<kMin>(), get<kMax>(), dataset.getData());
    } else {
      return Error(NoDataSet);
    }
    return {};
  }
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
      if (!mAlgorithm.initialized())
        return Error(NoDataFitted);
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
    if (!result.ok())
      return result;
    result = transform(sourceClient, destClient);
    return result;
  }

  MessageResult<void> transformPoint(BufferPtr in, BufferPtr out) {
    if (!mAlgorithm.initialized()) return Error(NoDataFitted);
    InOutBuffersCheck bufCheck(mAlgorithm.dims());
    if (!bufCheck.checkInputs(in.get(), out.get())) return Error(bufCheck.error());
    BufferAdaptor::Access outBuf(out.get());
    Result resizeResult = outBuf.resize(mDims, 1, outBuf.sampleRate());
    if (!resizeResult.ok()) return Error(BufferAlloc);
    RealVector src(mDims);
    RealVector dest(mDims);
    src = BufferAdaptor::ReadAccess(in.get()).samps(0, mDims, 0);
    mAlgorithm.setMin(get<kMin>());
    mAlgorithm.setMax(get<kMax>());
    mAlgorithm.processFrame(src, dest);
    outBuf.samps(0) = dest;
    return OK();
  }

  FLUID_DECLARE_MESSAGES(makeMessage("fit", &NormalizeClient::fit),
                         makeMessage("fitTransform",
                                     &NormalizeClient::fitTransform),
                         makeMessage("transform", &NormalizeClient::transform),
                         makeMessage("transformPoint",
                                     &NormalizeClient::transformPoint),
                         makeMessage("cols", &NormalizeClient::dims),
                         makeMessage("clear", &NormalizeClient::clear),
                         makeMessage("size", &NormalizeClient::size),
                         makeMessage("load", &NormalizeClient::load),
                         makeMessage("dump", &NormalizeClient::dump),
                         makeMessage("read", &NormalizeClient::read),
                         makeMessage("write", &NormalizeClient::write));

private:
  index mDims{0};
  FluidInputTrigger mTrigger;
};

using RTNormalizeClient = ClientWrapper<NormalizeClient>;

} // namespace client
} // namespace fluid
