#pragma once

#include "DataSetClient.hpp"
#include "LabelSetClient.hpp"
#include "DataSetErrorStrings.hpp"
#include "algorithms/KNNClassifier.hpp"
#include "algorithms/KNNRegressor.hpp"
#include "data/FluidDataSet.hpp"

#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/MessageSet.hpp>
#include <clients/common/OfflineClient.hpp>
#include <clients/common/ParameterSet.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/Result.hpp>
#include <clients/nrt/FluidNRTClientWrapper.hpp>
#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>
#include <string>

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

  MessageResult<std::string> index(DataSetClientRef datasetClient) const {
    auto datasetWeakPtr = datasetClient.get();
    if (auto datasetClientPtr = datasetWeakPtr.lock()) {
      auto dataset = datasetClientPtr->getDataSet();
      mTree = algorithm::KDTree{dataset};
    } else {
      return {Result::Status::kError, "DataSet doesn't exist"};
    }
    return {};
  }

  MessageResult<std::string> classify(BufferPtr data, LabelSetClientRef labelsetClient, int k) const {
    algorithm::KNNClassifier classifier;
    if (!data)
      return {Result::Status::kError, NoBufferError};
    if(mTree.nPoints() == 0)return {Result::Status::kError, "No index"};
    if (mTree.nPoints() < k)return {Result::Status::kError, "Not enough data in index"};
    auto labelsetPtr = labelsetClient.get().lock();
    if(!labelsetPtr)return {Result::Status::kError, "LabelSet doesn't exist"};
    auto labelSet = labelsetPtr->getLabelSet();
    BufferAdaptor::Access buf(data.get());
    if (buf.numFrames() != mTree.nDims())
        return {Result::Status::kError, WrongPointSizeError};
    FluidTensor<double, 1> point(mTree.nDims());
    point = buf.samps(0, mTree.nDims(), 0);
    std::string result = classifier.predict(mTree, point, labelSet, k);
    return result;
  }

  MessageResult<double> regress(BufferPtr data, DataSetClientRef targetDataSetClient, int k) const {
    algorithm::KNNRegressor regressor;
    if (!data)
      return {Result::Status::kError, NoBufferError};
    if(mTree.nPoints() == 0)return {Result::Status::kError, "No index"};
    else if (mTree.nPoints() < k)return {Result::Status::kError, "Not enough data in index"};
    auto tgtPtr = targetDataSetClient.get().lock();
    if(!tgtPtr)return {Result::Status::kError, "Target DataSet doesn't exist"};
    auto target = tgtPtr->getDataSet();
    BufferAdaptor::Access buf(data.get());
    if (buf.numFrames() != mTree.nDims()) return {Result::Status::kError, WrongPointSizeError};
    FluidTensor<double, 1> point(mTree.nDims());
    point = buf.samps(0, mTree.nDims(), 0);
    double result = regressor.predict(mTree, target, point, k);
    return result;
    }

  FLUID_DECLARE_MESSAGES(makeMessage("index", &KNNClient::index),
                         makeMessage("fit", &KNNClient::index)
                         makeMessage("classify", &KNNClient::classify),
                       makeMessage("regress", &KNNClient::regress));

private:
  mutable algorithm::KDTree mTree{0};
};

using NRTThreadedKNNClient = NRTThreadingAdaptor<ClientWrapper<KNNClient>>;

} // namespace client
} // namespace fluid
