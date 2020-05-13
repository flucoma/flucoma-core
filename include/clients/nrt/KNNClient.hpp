#pragma once

#include "NRTClient.hpp"
#include "algorithms/KNNClassifier.hpp"
#include "algorithms/KNNRegressor.hpp"

namespace fluid {
namespace client {

class KNNClient : public FluidBaseClient, OfflineIn, OfflineOut {
  enum { kNDims, kK };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using LabelSet = FluidDataSet<string, string, 1>;
  using DataSet = FluidDataSet<string, double, 1>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS();

  KNNClient(ParamSetViewType &p) : mParams(p) {}

  MessageResult<std::string> fit(DataSetClientRef datasetClient) {
    auto datasetWeakPtr = datasetClient.get();
    if (auto datasetClientPtr = datasetWeakPtr.lock()) {
      auto dataset = datasetClientPtr->getDataSet();
      mTree = algorithm::KDTree{dataset};
    } else {
      return NoDataSetError;
    }
    return {};
  }

  MessageResult<std::string> classify(BufferPtr data, LabelSetClientRef labelsetClient, fluid::index k) const {
    algorithm::KNNClassifier classifier;
    if (!data)
      return NoBufferError;
    if(k == 0) return SmallKError;
    if(mTree.nPoints() == 0)return NoDataFittedError;
    if (mTree.nPoints() < k)return NotEnoughDataError;
    auto labelsetPtr = labelsetClient.get().lock();
    if(!labelsetPtr) return NoLabelSetError;
    auto labelSet = labelsetPtr->getLabelSet();
    BufferAdaptor::Access buf(data.get());
    if (buf.numFrames() != mTree.nDims()) return WrongPointSizeError;
    FluidTensor<double, 1> point(mTree.nDims());
    point = buf.samps(0, mTree.nDims(), 0);
    std::string result = classifier.predict(mTree, point, labelSet, k);
    return result;
  }

  MessageResult<double> regress(BufferPtr data, DataSetClientRef targetDataSetClient, fluid::index k) const {
    algorithm::KNNRegressor regressor;
    if (!data)
      return NoBufferError;
    if(k == 0) return SmallKError;
    if(mTree.nPoints() == 0)return NoDataFittedError;
    else if (mTree.nPoints() < k)return NotEnoughDataError;
    auto tgtPtr = targetDataSetClient.get().lock();
    if(!tgtPtr)return NoDataSetError;
    auto target = tgtPtr->getDataSet();
    BufferAdaptor::Access buf(data.get());
    if (buf.numFrames() != mTree.nDims()) return WrongPointSizeError;
    FluidTensor<double, 1> point(mTree.nDims());
    point = buf.samps(0, mTree.nDims(), 0);
    double result = regressor.predict(mTree, target, point, k);
    return result;
    }

  FLUID_DECLARE_MESSAGES(makeMessage("fit", &KNNClient::fit),
                         makeMessage("classify", &KNNClient::classify),
                         makeMessage("classifyPoint", &KNNClient::classify),
                         makeMessage("regress", &KNNClient::regress),
                         makeMessage("regressPoint", &KNNClient::regress));

private:
  algorithm::KDTree mTree{0};
};

using NRTThreadedKNNClient = NRTThreadingAdaptor<ClientWrapper<KNNClient>>;

} // namespace client
} // namespace fluid
