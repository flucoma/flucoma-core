#pragma once

#include "NRTClient.hpp"
#include "algorithms/KNNClassifier.hpp"

namespace fluid {
namespace client {

class KNNClassifierClient : public FluidBaseClient, OfflineIn, OfflineOut {
  enum { kNDims, kK };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using LabelSet = FluidDataSet<string, string, 1>;
  using DataSet = FluidDataSet<string, double, 1>;
  using StringVector = FluidTensor<string, 1>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS();

  KNNClassifierClient(ParamSetViewType &p) : mParams(p) {}

  MessageResult<string> fit(
    DataSetClientRef datasetClient,
    LabelSetClientRef labelsetClient)
    {
    auto datasetClientPtr = datasetClient.get().lock();
    if(!datasetClientPtr) return Error<string>(NoDataSet);
    auto dataset = datasetClientPtr->getDataSet();
    if (dataset.size() == 0) return Error<string>(EmptyDataSet);
    auto labelsetPtr = labelsetClient.get().lock();
    if(!labelsetPtr) return Error<string>(NoLabelSet);
    auto labelSet = labelsetPtr->getLabelSet();
    if (labelSet.size() == 0) return Error<string>(EmptyLabelSet);
    if(dataset.size() != labelSet.size())
      return Error<string>("Different sizes for source and target");
    mTree = algorithm::KDTree{dataset};
    mLabels = labelSet;
    return {};
  }

  MessageResult<string> predictPoint(
    BufferPtr data, fluid::index k) const
    {
    algorithm::KNNClassifier classifier;
    if (!data) return Error<string>(NoBuffer);
    if(k == 0) return Error<string>(SmallK);
    if(mTree.nPoints() == 0) return Error<string>(NoDataFitted);
    if (mTree.nPoints() < k) return Error<string>(NotEnoughData);
    BufferAdaptor::Access buf(data.get());
    if (buf.numFrames() != mTree.nDims()) return Error<string>(WrongPointSize);

    RealVector point(mTree.nDims());
    point = buf.samps(0, mTree.nDims(), 0);
    std::string result = classifier.predict(mTree, point, mLabels, k);
    return result;
  }

  MessageResult<void> predict(
    DataSetClientRef source,
    LabelSetClientRef dest, fluid::index k) const
    {
    auto sourcePtr = source.get().lock();
    if(!sourcePtr) return Error(NoDataSet);
    auto dataSet = sourcePtr->getDataSet();
    if (dataSet.size() == 0) return Error(EmptyDataSet);
    auto destPtr = dest.get().lock();
    if(!destPtr) return Error(NoLabelSet);
    if (dataSet.pointSize()!=mTree.nDims()) return Error(WrongPointSize);
    if(k == 0) return Error(SmallK);
    if(mTree.nPoints() == 0) return Error(NoDataFitted);
    if (mTree.nPoints() < k) return Error(NotEnoughData);

    algorithm::KNNClassifier classifier;
    auto ids = dataSet.getIds();
    auto data = dataSet.getData();
    LabelSet result(1);
    for (index i = 0; i < dataSet.size(); i++) {
      RealVectorView point = data.row(i);
      StringVector label = {
        classifier.predict(mTree, point, mLabels, k)
      };
      result.add(ids(i), label);
    }
    destPtr->setLabelSet(result);
    return OK();
  }

  FLUID_DECLARE_MESSAGES(makeMessage("fit",
                         &KNNClassifierClient::fit),
                         makeMessage("predict",
                         &KNNClassifierClient::predict),
                         makeMessage("predictPoint",
                         &KNNClassifierClient::predictPoint)
                       );

private:
  algorithm::KDTree mTree{0};
  LabelSet mLabels{1};
};

using NRTThreadedKNNClassifierClient = NRTThreadingAdaptor<ClientWrapper<KNNClassifierClient>>;

} // namespace client
} // namespace fluid
