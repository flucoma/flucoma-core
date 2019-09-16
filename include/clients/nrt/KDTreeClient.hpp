#pragma once

#include "DatasetClient.hpp"
#include "DatasetErrorStrings.hpp"
#include "algorithms/KDTree.hpp"
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

class KDTreeClient : public FluidBaseClient, OfflineIn, OfflineOut {
  enum { kNDims };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS(LongParam<Fixed<true>>("nDims", "Dimension size", 1,
                                              Min(1)));

  KDTreeClient(ParamSetViewType &p) : mParams(p), mDataset(get<kNDims>()) {
    mDims = get<kNDims>();
  }

  MessageResult<void> index(DatasetClientRef datasetClient) {
    auto weakPtr = datasetClient.get();
    if (auto datasetClientPtr = weakPtr.lock()) {
      auto dataset = datasetClientPtr->getDataset();
      if (dataset.size() == 0) return {Result::Status::kError, EmptyDatasetError};
      mTree = algorithm::KDTree<string>(dataset);
    } else {
      return {Result::Status::kError, "Dataset doesn't exist"};
    }
    return {Result::Status::kOk};
  }

  MessageResult<FluidTensor<std::string, 1>> knn(BufferPtr data, int k) const {

    if (!data)
      return {Result::Status::kError, NoBufferError};
    BufferAdaptor::Access buf(data.get());
    if (buf.numFrames() != mDims)
      return {Result::Status::kError, WrongSizeError};
    if (k > mTree.nPoints()){
      return {Result::Status::kError, SmallDatasetError};
    }
    if(k <= 0 ){
      return {Result::Status::kError, "k should be at least 1"};
    }

    FluidTensor<double, 1> point(mDims);
    point = buf.samps(0, mDims, 0);
    FluidDataset<int, double, std::string, 1> nearest =
        mTree.kNearest(point, k);

    FluidTensor<std::string, 1> result{nearest.getTargets()};
    return result;
  }

  FLUID_DECLARE_MESSAGES(makeMessage("index", &KDTreeClient::index),
                         makeMessage("knn", &KDTreeClient::knn));

private:
  mutable FluidDataset<string, double, string, 1> mDataset;
  mutable algorithm::KDTree<string> mTree{1};
  size_t mDims;
};

using NRTThreadedKDTreeClient =
    NRTThreadingAdaptor<ClientWrapper<KDTreeClient>>;

} // namespace client
} // namespace fluid
