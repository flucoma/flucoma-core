#pragma once

#include "DataSetClient.hpp"
#include "LabelSetClient.hpp"
#include "NRTClient.hpp"
#include "algorithms/KMeans.hpp"
#include <string>

namespace fluid {
namespace client {

class KMeansClient : public FluidBaseClient,
                     OfflineIn,
                     OfflineOut,
                     ModelObject,
                     public DataClient<algorithm::KMeans> {

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using IndexVector = FluidTensor<index, 1>;
  using StringVector = FluidTensor<string, 1>;
  using StringVectorView = FluidTensorView<string, 1>;
  using LabelSet = FluidDataSet<string, string, 1>;
  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS();

  KMeansClient(ParamSetViewType &p) : DataClient(mModel), mParams(p) {}

  MessageResult<IndexVector> fit(DataSetClientRef datasetClient, index k,
                                 index maxIter) {
    auto datasetClientPtr = datasetClient.get().lock();
    if (!datasetClientPtr)
      return Error<IndexVector>(NoDataSet);
    auto dataSet = datasetClientPtr->getDataSet();
    if (dataSet.size() == 0)
      return Error<IndexVector>(EmptyDataSet);
    if (k <= 1)
      return Error<IndexVector>(SmallK);
    if (maxIter <= 0)
      maxIter = 100;
    mModel.init(k, dataSet.dims());
    mModel.train(dataSet, maxIter);
    IndexVector assignments(dataSet.size());
    mModel.getAssignments(assignments);
    return getCounts(assignments, k);
  }

  MessageResult<IndexVector> fitPredict(DataSetClientRef datasetClient,
                                        LabelSetClientRef labelsetClient,
                                        index k, index maxIter) {
    auto datasetClientPtr = datasetClient.get().lock();
    if (!datasetClientPtr)
      return Error<IndexVector>(NoDataSet);
    auto dataSet = datasetClientPtr->getDataSet();
    if (dataSet.size() == 0)
      return Error<IndexVector>(EmptyDataSet);
    auto labelsetClientPtr = labelsetClient.get().lock();
    if (!labelsetClientPtr)
      return Error<IndexVector>(NoLabelSet);
    if (k <= 1)
      return Error<IndexVector>(SmallK);
    if (maxIter <= 0)
      maxIter = 100;
    mK = k;
    mModel.init(k, dataSet.pointSize());
    mModel.train(dataSet, maxIter);
    IndexVector assignments(dataSet.size());
    mModel.getAssignments(assignments);
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
    if (!mModel.trained())
      return Error<IndexVector>(NoDataFitted);
    if (dataSet.dims() != mModel.dims())
      return Error<IndexVector>(WrongPointSize);
    StringVectorView ids = dataSet.getIds();
    IndexVector assignments(dataSet.size());
    RealVector query(mModel.dims());
    for (index i = 0; i < dataSet.size(); i++) {
      dataSet.get(ids(i), query);
      assignments(i) = mModel.vq(query);
    }
    labelsetClientPtr->setLabelSet(getLabels(ids, assignments));
    return getCounts(assignments, mModel.getK());
  }

  MessageResult<index> predictPoint(BufferPtr data) const {
    if (!data)
      return Error<index>(NoBuffer);
    if (!mModel.trained())
      return Error<index>(NoDataFitted);
    BufferAdaptor::Access buf(data.get());
    if (!buf.exists())
      return Error<index>(InvalidBuffer);
    if (buf.numFrames() != mModel.dims())
      return Error<index>(WrongPointSize);
    RealVector point(mModel.dims());
    point = buf.samps(0, mModel.dims(), 0);
    return mModel.vq(point);
  }

  FLUID_DECLARE_MESSAGES(makeMessage("fit", &KMeansClient::fit),
                         makeMessage("predict", &KMeansClient::predict),
                         makeMessage("predictPoint",
                                     &KMeansClient::predictPoint),
                         makeMessage("fitPredict", &KMeansClient::fitPredict),
                         makeMessage("cols", &KMeansClient::dims),
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

  algorithm::KMeans mModel;
  index mK{0};
};

using NRTThreadedKMeansClient =
    NRTThreadingAdaptor<ClientWrapper<KMeansClient>>;

} // namespace client
} // namespace fluid
