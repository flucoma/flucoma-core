//modified version of NormalizeClient.hpp code
#pragma once

#include "DataSetClient.hpp"
#include "NRTClient.hpp"
#include "algorithms/RobustScaling.hpp"

namespace fluid {
namespace client {

class RobustScaleClient : public FluidBaseClient,
                        AudioIn,
                        ControlOut,
                        ModelObject,
                        public DataClient<algorithm::RobustScaling> {

  enum { kLow, kHigh, kInvert, kInputBuffer, kOutputBuffer };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using StringVector = FluidTensor<string, 1>;

  FLUID_DECLARE_PARAMS(FloatParam("low", "Low Percentile", 0, Min(0), Max(100)),
                       FloatParam("high", "High Percentile", 100, Min(0), Max(100)),
                       EnumParam("invert", "Inverse Transform", 0, "False", "True"),
                       BufferParam("inputPointBuffer", "Input Point Buffer"),
                       BufferParam("predictionBuffer", "Prediction Buffer"));

  RobustScaleClient(ParamSetViewType &p) : mParams(p) {
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
    auto outBuf = BufferAdaptor::Access(get<kOutputBuffer>().get());
    if(outBuf.samps(0).size() != mAlgorithm.dims()) return;
    RealVector src(mAlgorithm.dims());
    RealVector dest(mAlgorithm.dims());
    src = BufferAdaptor::ReadAccess(get<kInputBuffer>().get()).samps(0, mAlgorithm.dims(), 0);
    mAlgorithm.setLow(get<kLow>());
    mAlgorithm.setHigh(get<kHigh>());
    mTrigger.process(input, output, [&]() {
      mAlgorithm.processFrame(src, dest, get<kInvert>() == 1);
      outBuf.samps(0) = dest;
    });
  }

  index latency() { return 0; }

  MessageResult<void> fit(DataSetClientRef datasetClient) {
    auto weakPtr = datasetClient.get();
    if (auto datasetClientPtr = weakPtr.lock()) {
      auto dataset = datasetClientPtr->getDataSet();
      if (dataset.size() == 0)
        return Error(EmptyDataSet);
      mAlgorithm.init(get<kLow>(), get<kHigh>(), dataset.getData());
    } else {
      return Error(NoDataSet);
    }
    return {};
  }
  MessageResult<void> transform(DataSetClientRef sourceClient,
                                DataSetClientRef destClient) {
    return _transform(sourceClient, destClient, get<kInvert>() == 1);
  }

  MessageResult<void> fitTransform(DataSetClientRef sourceClient,
                                   DataSetClientRef destClient) {
    auto result = fit(sourceClient);
    if (!result.ok())
      return result;
    result = _transform(sourceClient, destClient, false);
    return result;
  }

  MessageResult<void> transformPoint(BufferPtr in, BufferPtr out) {
    if (!mAlgorithm.initialized()) return Error(NoDataFitted);
    InOutBuffersCheck bufCheck(mAlgorithm.dims());
    if (!bufCheck.checkInputs(in.get(), out.get())) return Error(bufCheck.error());
    BufferAdaptor::Access outBuf(out.get());
    Result resizeResult = outBuf.resize(mAlgorithm.dims(), 1, outBuf.sampleRate());
    if (!resizeResult.ok()) return Error(BufferAlloc);
    RealVector src(mAlgorithm.dims());
    RealVector dest(mAlgorithm.dims());
    src = BufferAdaptor::ReadAccess(in.get()).samps(0, mAlgorithm.dims(), 0);
    mAlgorithm.setLow(get<kLow>());
    mAlgorithm.setHigh(get<kHigh>());
    mAlgorithm.processFrame(src, dest, get<kInvert>() == 1);
    outBuf.samps(0) = dest;
    return OK();
  }

  FLUID_DECLARE_MESSAGES(makeMessage("fit", &RobustScaleClient::fit),
                         makeMessage("fitTransform",
                                     &RobustScaleClient::fitTransform),
                         makeMessage("transform", &RobustScaleClient::transform),
                         makeMessage("transformPoint",
                                     &RobustScaleClient::transformPoint),
                         makeMessage("cols", &RobustScaleClient::dims),
                         makeMessage("clear", &RobustScaleClient::clear),
                         makeMessage("size", &RobustScaleClient::size),
                         makeMessage("load", &RobustScaleClient::load),
                         makeMessage("dump", &RobustScaleClient::dump),
                         makeMessage("read", &RobustScaleClient::read),
                         makeMessage("write", &RobustScaleClient::write));

private:
  MessageResult<void> _transform(DataSetClientRef sourceClient,
                                DataSetClientRef destClient, bool invert) {
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
      mAlgorithm.setLow(get<kLow>());
      mAlgorithm.setHigh(get<kHigh>());
      mAlgorithm.process(srcDataSet.getData(), data, invert);
      FluidDataSet<string, double, 1> result(ids, data);
      destPtr->setDataSet(result);
    } else {
      return Error(NoDataSet);
    }
    return OK();
  }
  FluidInputTrigger mTrigger;
};

using RTRobustScaleClient = ClientWrapper<RobustScaleClient>;

} // namespace client
} // namespace fluid
