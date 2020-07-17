#pragma once

#include "DataSetClient.hpp"
#include "LabelSetClient.hpp"
#include "NRTClient.hpp"
#include "algorithms/KMeans.hpp"
#include <string>

namespace fluid {
namespace client {

class KMeansClient : public FluidBaseClient,
                     AudioIn,
                     ControlOut,
                     ModelObject,
                     public DataClient<algorithm::KMeans> {

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using IndexVector = FluidTensor<index, 1>;
  using StringVector = FluidTensor<string, 1>;
  using StringVectorView = FluidTensorView<string, 1>;
  using LabelSet = FluidDataSet<string, string, 1>;

  enum {kNumClusters, kMaxIter, kInputBuffer, kOutputBuffer};

  FLUID_DECLARE_PARAMS(
      LongParam("numClusters","Number of Clusters", 4, Min(1)),
      LongParam("maxIter","Max number of Iterations", 100, Min(1)),
      BufferParam("inputPointBuffer","Input Point Buffer"),
      BufferParam("predictionBuffer","Prediction Buffer")
  );

  KMeansClient(ParamSetViewType &p) : mParams(p)
  {
    audioChannelsIn(1);
    controlChannelsOut(1);
  }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>> &input,
               std::vector<FluidTensorView<T, 1>> &output, FluidContext &)
  {
    if (!mAlgorithm.trained()) return;
    InOutBuffersCheck bufCheck(mAlgorithm.dims());
    if(!bufCheck.checkInputs(
      get<kInputBuffer>().get(),
      get<kOutputBuffer>().get()))
      return;
    RealVector point(mAlgorithm.dims());
    point = BufferAdaptor::ReadAccess(get<kInputBuffer>().get()).samps(0, mAlgorithm.dims(), 0);
    mTrigger.process(input, output, [&](){
       BufferAdaptor::Access(get<kOutputBuffer>().get()).samps(0)[0] = mAlgorithm.vq(point);
    });
  }

  MessageResult<IndexVector> fit(DataSetClientRef datasetClient) {
    index k = get<kNumClusters>();
    index maxIter = get<kMaxIter>();
    auto datasetClientPtr = datasetClient.get().lock();
    if (!datasetClientPtr) return Error<IndexVector>(NoDataSet);
    auto dataSet = datasetClientPtr->getDataSet();
    if (dataSet.size() == 0) return Error<IndexVector>(EmptyDataSet);
    if (k <= 1) return Error<IndexVector>(SmallK);
    mAlgorithm.init(k, dataSet.dims());
    mAlgorithm.train(dataSet, maxIter);
    IndexVector assignments(dataSet.size());
    mAlgorithm.getAssignments(assignments);
    return getCounts(assignments, k);
  }

  MessageResult<IndexVector> fitPredict(DataSetClientRef datasetClient,
                                        LabelSetClientRef labelsetClient) {
    index k = get<kNumClusters>();
    index maxIter = get<kMaxIter>();
    auto datasetClientPtr = datasetClient.get().lock();
    if (!datasetClientPtr)
      return Error<IndexVector>(NoDataSet);
    auto dataSet = datasetClientPtr->getDataSet();
    if (dataSet.size() == 0)
      return Error<IndexVector>(EmptyDataSet);
    auto labelsetClientPtr = labelsetClient.get().lock();
    if (!labelsetClientPtr)
      return Error<IndexVector>(NoLabelSet);
    if (k <= 1) return Error<IndexVector>(SmallK);
    if (maxIter <= 0) maxIter = 100;
    mAlgorithm.init(k, dataSet.pointSize());
    mAlgorithm.train(dataSet, maxIter);
    IndexVector assignments(dataSet.size());
    mAlgorithm.getAssignments(assignments);
    StringVectorView ids = dataSet.getIds();
    labelsetClientPtr->setLabelSet(getLabels(ids, assignments));
    return getCounts(assignments, k);
  }

  MessageResult<IndexVector> predict(DataSetClientRef datasetClient,
                                     LabelSetClientRef labelClient) const {
    auto dataPtr = datasetClient.get().lock();
    if (!dataPtr)
      return Error<IndexVector>(NoDataSet);
    auto labelsetClientPtr = labelClient.get().lock();
    if (!labelsetClientPtr)
      return Error<IndexVector>(NoLabelSet);
    auto dataSet = dataPtr->getDataSet();
    if (dataSet.size() == 0)
      return Error<IndexVector>(EmptyDataSet);
    if (!mAlgorithm.trained())
      return Error<IndexVector>(NoDataFitted);
    if (dataSet.dims() != mAlgorithm.dims())
      return Error<IndexVector>(WrongPointSize);
    StringVectorView ids = dataSet.getIds();
    IndexVector assignments(dataSet.size());
    RealVector query(mAlgorithm.dims());
    for (index i = 0; i < dataSet.size(); i++) {
      dataSet.get(ids(i), query);
      assignments(i) = mAlgorithm.vq(query);
    }
    labelsetClientPtr->setLabelSet(getLabels(ids, assignments));
    return getCounts(assignments, mAlgorithm.getK());
  }
  MessageResult<index> predictPoint(BufferPtr data) const {
    if (!mAlgorithm.trained()) return Error<index>(NoDataFitted);
    InBufferCheck bufCheck(mAlgorithm.dims());
    if(!bufCheck.checkInputs(data.get())) return Error<index>(bufCheck.error());
    RealVector point(mAlgorithm.dims());
    point = BufferAdaptor::ReadAccess(data.get()).samps(0, mAlgorithm.dims(), 0);
    return mAlgorithm.vq(point);
  }

  index latency() { return 0; }

  FLUID_DECLARE_MESSAGES(makeMessage("fit", &KMeansClient::fit),
                         makeMessage("predict", &KMeansClient::predict),
                         makeMessage("predictPoint",
                                     &KMeansClient::predictPoint),
                         makeMessage("fitPredict", &KMeansClient::fitPredict),
                         makeMessage("cols", &KMeansClient::dims),
                         makeMessage("clear", &KMeansClient::clear),
                         makeMessage("size", &KMeansClient::size),
                         makeMessage("load", &KMeansClient::load),
                         makeMessage("dump", &KMeansClient::dump),
                         makeMessage("write", &KMeansClient::write),
                         makeMessage("read", &KMeansClient::read));

private:
  IndexVector getCounts(IndexVector assignments, index k) const {
    IndexVector counts(k);
    counts.fill(0);
    for (auto a : assignments)
      counts[a]++;
    return counts;
  }

  LabelSet getLabels(StringVectorView &ids, IndexVector assignments) const {
    LabelSet result(1);
    for (index i = 0; i < ids.size(); i++) {
      StringVector point = {std::to_string(assignments(i))};
      result.add(ids(i), point);
    }
    return result;
  }

  FluidInputTrigger mTrigger;
};

using RTKMeansClient = ClientWrapper<KMeansClient>;

} // namespace client
} // namespace fluid
