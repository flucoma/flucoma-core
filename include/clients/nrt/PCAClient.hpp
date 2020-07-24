#pragma once
#include "DataSetClient.hpp"
#include "NRTClient.hpp"
#include "algorithms/PCA.hpp"

namespace fluid {
namespace client {

class PCAClient : public FluidBaseClient,
                  AudioIn,
                  ControlOut,
                  ModelObject,
                  public DataClient<algorithm::PCA> {
public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using StringVector = FluidTensor<string, 1>;

  enum { kNumDimensions, kInputBuffer, kOutputBuffer };

  FLUID_DECLARE_PARAMS(
    LongParam("numDimensions", "Target Number of Dimensions", 2, Min(1)),
    BufferParam("inputPointBuffer", "Input Point Buffer"),
                       BufferParam("predictionBuffer", "Prediction Buffer"));

  PCAClient(ParamSetViewType &p) : mParams(p)
                       {
                         audioChannelsIn(1);
                         controlChannelsOut(1);
                       }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>> &input,
               std::vector<FluidTensorView<T, 1>> &output, FluidContext &) {
    if (!mAlgorithm.initialized())
      return;
    InOutBuffersCheck bufCheck(mAlgorithm.dims());
    if (!bufCheck.checkInputs(
        get<kInputBuffer>().get(),
        get<kOutputBuffer>().get())) return;
    auto outBuf = BufferAdaptor::Access(get<kOutputBuffer>().get());
    if(outBuf.samps(0).size() != mAlgorithm.size()) return;

    RealVector src(mAlgorithm.dims());
    RealVector dest(mAlgorithm.size());
    src = BufferAdaptor::ReadAccess(get<kInputBuffer>().get()).samps(0, mAlgorithm.dims(), 0);
    mTrigger.process(input, output, [&]() {
      mAlgorithm.processFrame(src, dest);
      outBuf.samps(0) = dest;
    });
  }

  MessageResult<void> fit(DataSetClientRef datasetClient) {
    index k = get<kNumDimensions>();
    auto datasetClientPtr = datasetClient.get().lock();
    if (!datasetClientPtr) return Error(NoDataSet);
    auto dataSet = datasetClientPtr->getDataSet();
    if (dataSet.size() == 0) return Error(EmptyDataSet);
    if (k <= 0) return Error(SmallK);
    if (dataSet.pointSize() < k)
      return Error("k is larger than the current dimensions");
    if (k > dataSet.size() - 1) return Error(NotEnoughData);

    mAlgorithm.init(dataSet.getData(), k);
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
    if (!mAlgorithm.initialized()) return Error(NoDataFitted);
    InOutBuffersCheck bufCheck(mAlgorithm.dims());
    if (!bufCheck.checkInputs(in.get(), out.get())) return Error(bufCheck.error());
    BufferAdaptor::Access outBuf(out.get());
    Result resizeResult = outBuf.resize(mAlgorithm.size(), 1, outBuf.sampleRate());
    if (!resizeResult.ok()) return Error(BufferAlloc);
    FluidTensor<double, 1> src(mAlgorithm.dims());
    FluidTensor<double, 1> dest(mAlgorithm.size());
    src = BufferAdaptor::ReadAccess(in.get()).samps(0, mAlgorithm.dims(), 0);
    mAlgorithm.processFrame(src, dest);
    BufferAdaptor::Access(out.get()).samps(0) = dest;
    return {};
  }

  index latency() { return 0; }

  FLUID_DECLARE_MESSAGES(makeMessage("fit", &PCAClient::fit),
                         makeMessage("transform", &PCAClient::transform),
                         makeMessage("fitTransform", &PCAClient::fitTransform),
                         makeMessage("transformPoint",
                                     &PCAClient::transformPoint),
                         makeMessage("cols", &PCAClient::dims),
                         makeMessage("size", &PCAClient::size),
                         makeMessage("clear", &PCAClient::clear),
                         makeMessage("load", &PCAClient::load),
                         makeMessage("dump", &PCAClient::dump),
                         makeMessage("read", &PCAClient::read),
                         makeMessage("write", &PCAClient::write));
private:
  FluidInputTrigger mTrigger;
};

using RTPCAClient = ClientWrapper<PCAClient>;
} // namespace client
} // namespace fluid
