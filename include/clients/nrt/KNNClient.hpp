#pragma once

#include "DatasetClient.hpp"
#include "DatasetErrorStrings.hpp"
#include "algorithms/KNNClassifier.hpp"
#include "data/FluidDataset.hpp"

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

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS(
    //LongParam<Fixed<true>>("nDims", "Dimension size", 1,
      //                                        Min(1))
  );

  KNNClient(ParamSetViewType &p) : mParams(p) {}

  MessageResult<std::string> index(DatasetClientRef datasetClient) const {
    auto weakPtr = datasetClient.get();
    if (auto datasetClientPtr = weakPtr.lock()) {
      auto dataset = datasetClientPtr->getDataset();
      //dataset.print();
      mTree = algorithm::KDTree<std::string>{dataset};
    } else {
      return {Result::Status::kError, "Dataset doesn't exist"};
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
        return {Result::Status::kError, WrongSizeError};
      FluidTensor<double, 1> point(mTree.nDims());
      point = buf.samps(0, mTree.nDims(), 0);
      std::string result = classifier.predict(mTree, point, k);
      return result;
    }
  }


  FLUID_DECLARE_MESSAGES(makeMessage("index", &KNNClient::index),
                         makeMessage("classify", &KNNClient::classify));

private:
  //mutable FluidDataset<string, double, string, 1> mDataset;
  mutable algorithm::KDTree<std::string> mTree{0};
  //size_t mDims;
};

using NRTThreadedKNNClient = NRTThreadingAdaptor<ClientWrapper<KNNClient>>;

} // namespace client
} // namespace fluid
