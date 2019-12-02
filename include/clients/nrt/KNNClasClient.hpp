#pragma once

#include "DataSetClient.hpp"
#include "LabelSetClient.hpp"
#include "DataSetErrorStrings.hpp"
#include "algorithms/KNNClassifier.hpp"
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

class KNNClasClient : public FluidBaseClient, OfflineIn, OfflineOut {
  enum { kNDims, kK };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using LabelSet = FluidDataSet<string, string, 1>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS();

  KNNClasClient(ParamSetViewType &p) : mParams(p) {}

  MessageResult<std::string> index(DataSetClientRef datasetClient,
    LabelSetClientRef labelsetClient) const {
    auto datasetWeakPtr = datasetClient.get();
    if (auto datasetClientPtr = datasetWeakPtr.lock()) {
      auto dataset = datasetClientPtr->getDataSet();
      mTree = algorithm::KDTree{dataset};
    } else {
      return {Result::Status::kError, "DataSet doesn't exist"};
    }

    auto labelsetWeakPtr = labelsetClient.get();
    if (auto labelsetClientPtr = labelsetWeakPtr.lock()) {
      mLabelSet = labelsetClientPtr->getLabelSet();
    } else {
      return {Result::Status::kError, "LabelSet doesn't exist"};
    }

    return {};
  }

  MessageResult<std::string> classify(BufferPtr data, int k) const {
    algorithm::KNNClassifier classifier;
    if (!data)
      return {Result::Status::kError, NoBufferError};
    if(mTree.nPoints() == 0)return {Result::Status::kError, "No index"};
    else if (mTree.nPoints() < k)return {Result::Status::kError, "Not enough data in index"};
    else{
      BufferAdaptor::Access buf(data.get());
      if (buf.numFrames() != mTree.nDims())
        return {Result::Status::kError, WrongPointSizeError};
      FluidTensor<double, 1> point(mTree.nDims());
      point = buf.samps(0, mTree.nDims(), 0);
      std::string result = classifier.predict(mTree, point, mLabelSet, k);
      return result;
    }
  }

  FLUID_DECLARE_MESSAGES(makeMessage("index", &KNNClasClient::index),
                         makeMessage("fit", &KNNClasClient::index),
                         makeMessage("classifyPoint", &KNNClasClient::classify),
                         makeMessage("classify", &KNNClasClient::classify));

private:
  mutable algorithm::KDTree mTree{0};
  mutable LabelSet mLabelSet{1};
};

using NRTThreadedKNNClasClient = NRTThreadingAdaptor<ClientWrapper<KNNClasClient>>;

} // namespace client
} // namespace fluid
