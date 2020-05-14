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

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS();

  KNNClassifierClient(ParamSetViewType &p) : mParams(p) {}

  MessageResult<std::string> fit(
    DataSetClientRef datasetClient,
    LabelSetClientRef labelsetClient)
    {
    auto datasetClientPtr = datasetClient.get().lock();
    if(!datasetClientPtr) return NoDataSetError;
    auto dataset = datasetClientPtr->getDataSet();
    if (dataset.size() == 0) return EmptyDataSetError;
    auto labelsetPtr = labelsetClient.get().lock();
    if(!labelsetPtr) return NoLabelSetError;
    auto labelSet = labelsetPtr->getLabelSet();
    if (labelSet.size() == 0) return EmptyLabelSetError;
    if(dataset.size() != labelSet.size())
      return {Result::Status::kError, "Different sizes for source and target"};
    mTree = algorithm::KDTree{dataset};
    mLabels = labelSet;
    return OKResult;
  }

  MessageResult<std::string> predictPoint(
    BufferPtr data, fluid::index k) const
    {
    algorithm::KNNClassifier classifier;
    if (!data) return NoBufferError;
    if(k == 0) return SmallKError;
    if(mTree.nPoints() == 0) return NoDataFittedError;
    if (mTree.nPoints() < k) return NotEnoughDataError;
    BufferAdaptor::Access buf(data.get());
    if (buf.numFrames() != mTree.nDims()) return WrongPointSizeError;

    FluidTensor<double, 1> point(mTree.nDims());
    point = buf.samps(0, mTree.nDims(), 0);
    std::string result = classifier.predict(mTree, point, mLabels, k);
    return result;
  }

  MessageResult<void> predict(
    DataSetClientRef source,
    LabelSetClientRef dest, fluid::index k) const
    {
    auto sourcePtr = source.get().lock();
    if(!sourcePtr) return NoDataSetError;
    auto dataSet = sourcePtr->getDataSet();
    if (dataSet.size() == 0) return EmptyDataSetError;
    auto destPtr = dest.get().lock();
    if(!destPtr) return NoLabelSetError;
    if (dataSet.pointSize()!=mTree.nDims()) return WrongPointSizeError;
    if(k == 0) return SmallKError;
    if(mTree.nPoints() == 0) return NoDataFittedError;
    if (mTree.nPoints() < k) return NotEnoughDataError;

    algorithm::KNNClassifier classifier;
    auto ids = dataSet.getIds();
    auto data = dataSet.getData();
    LabelSet result(1);
    for (index i = 0; i < dataSet.size(); i++) {
      FluidTensorView<double, 1> point = data.row(i);
      FluidTensor<string, 1> label = {
        classifier.predict(mTree, point, mLabels, k)
      };
      result.add(ids(i), label);
    }
    destPtr->setLabelSet(result);
    return OKResult;
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
