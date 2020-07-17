#pragma once

#include "DataSetClient.hpp"
#include "NRTClient.hpp"
#include "algorithms/KDTree.hpp"
#include <string>

namespace fluid {
namespace client {

class KDTreeClient : public FluidBaseClient,
                     AudioIn,
                     ControlOut,
                     ModelObject,
                     public DataClient<algorithm::KDTree> {
public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using StringVector = FluidTensor<string, 1>;

  enum { kNumNeighbors, kDataSet, kInputBuffer, kOutputBuffer };

  FLUID_DECLARE_PARAMS(LongParam("numNeighbours", "Number of Nearest Neighbours",
                                 1),
                       DataSetClientRef::makeParam("dataSet", "DataSet Name"),
                       BufferParam("inputPointBuffer", "Input Point Buffer"),
                       BufferParam("predictionBuffer", "Prediction Buffer"));
  KDTreeClient(ParamSetViewType &p) : mParams(p)
  {
    audioChannelsIn(1);
    controlChannelsOut(1);
  }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>> &input,
               std::vector<FluidTensorView<T, 1>> &output, FluidContext &) {
    index k = get<kNumNeighbors>();
    if (!mAlgorithm.initialized()) return;
    if (k > mAlgorithm.size() || k <= 0) return;
    InOutBuffersCheck bufCheck(mAlgorithm.dims());
    if (!bufCheck.checkInputs(get<kInputBuffer>().get(),
                              get<kOutputBuffer>().get())) return;
    auto datasetClientPtr = get<kDataSet>().get().lock();
    if (!datasetClientPtr) datasetClientPtr = mDataSetClient.get().lock();
    if (!datasetClientPtr) return;
    auto dataset = datasetClientPtr->getDataSet();
    index pointSize = dataset.pointSize();
    mTrigger.process(input, output, [&]() {
      RealVector point(mAlgorithm.dims());
      point = BufferAdaptor::ReadAccess(get<kInputBuffer>().get())
        .samps(0, mAlgorithm.dims(), 0);
      if(mRTBuffer.size() != k * pointSize){
        mRTBuffer = RealVector(k * pointSize);
        mRTBuffer.fill(0);
      }
      auto nearest = mAlgorithm.kNearest(point, k);
      auto ids = nearest.getIds();
      for (index i = 0; i < k; i++) {
        dataset.get(ids(i), mRTBuffer(Slice(i * pointSize, pointSize)));
      }
      BufferAdaptor::Access(get<kOutputBuffer>().get()).samps(0) = mRTBuffer;
    });
  }

  index latency() { return 0; }

  MessageResult<void> fit(DataSetClientRef datasetClient) {
    mDataSetClient = datasetClient;
    auto datasetClientPtr = mDataSetClient.get().lock();
    if (!datasetClientPtr)
      return Error(NoDataSet);
    auto dataset = datasetClientPtr->getDataSet();
    if (dataset.size() == 0)
      return Error(EmptyDataSet);
    mAlgorithm = algorithm::KDTree(dataset);
    return OK();
  }

  MessageResult<StringVector> kNearest(BufferPtr data) const {
    index k = get<kNumNeighbors>();
    if (k > mAlgorithm.size()) return Error<StringVector>(SmallDataSet);
    if (k <= 0) return Error<StringVector>(SmallK);
    if (!mAlgorithm.initialized()) return Error<StringVector>(NoDataFitted);
    InBufferCheck bufCheck(mAlgorithm.dims());
    if (!bufCheck.checkInputs(data.get()))
        return Error<StringVector>(bufCheck.error());
    RealVector point(mAlgorithm.dims());
    point = BufferAdaptor::ReadAccess(data.get()).samps(0, mAlgorithm.dims(), 0);
    FluidDataSet<std::string, double, 1> nearest =
        mAlgorithm.kNearest(point, k);
    StringVector result{nearest.getIds()};
    return result;
  }

  MessageResult<RealVector> kNearestDist(BufferPtr data) const {
    // TODO: refactor with kNearest
    index k = get<kNumNeighbors>();
    if (k > mAlgorithm.size()) return Error<RealVector>(SmallDataSet);
    if (k <= 0) return Error<RealVector>(SmallK);
    if (!mAlgorithm.initialized()) return Error<RealVector>(NoDataFitted);
    InBufferCheck bufCheck(mAlgorithm.dims());
    if (!bufCheck.checkInputs(data.get()))
      return Error<RealVector>(bufCheck.error());
    RealVector point(mAlgorithm.dims());
    point = BufferAdaptor::ReadAccess(data.get()).samps(0, mAlgorithm.dims(), 0);
    FluidDataSet<std::string, double, 1> nearest =
        mAlgorithm.kNearest(point, k);
    RealVector result{nearest.getData().col(0)};
    return result;
  }

  FLUID_DECLARE_MESSAGES(makeMessage("fit", &KDTreeClient::fit),
                         makeMessage("kNearest", &KDTreeClient::kNearest),
                         makeMessage("kNearestDist",
                                     &KDTreeClient::kNearestDist),
                         makeMessage("cols", &KDTreeClient::dims),
                         makeMessage("clear", &KDTreeClient::clear),
                         makeMessage("size", &KDTreeClient::size),
                         makeMessage("load", &KDTreeClient::load),
                         makeMessage("dump", &KDTreeClient::dump),
                         makeMessage("write", &KDTreeClient::write),
                         makeMessage("read", &KDTreeClient::read));

private:
  FluidInputTrigger mTrigger;
  RealVector mRTBuffer;
  DataSetClientRef mDataSetClient;
};

using RTKDTreeClient = ClientWrapper<KDTreeClient>;

} // namespace client
} // namespace fluid
