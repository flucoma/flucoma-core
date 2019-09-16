#pragma once

//#include "DatasetClient.hpp"
#include "DatasetErrorStrings.hpp"
#include "algorithms/KNNRegressor.hpp"
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

class KNNRegClient : public FluidBaseClient, OfflineIn, OfflineOut {
  enum { kNDims, kK };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS();

  KNNRegClient(ParamSetViewType &p) : mParams(p) {}

  MessageResult<std::string> index(DatasetClientRef datasetClient, BufferPtr target) const {
    auto weakPtr = datasetClient.get();
    if (auto datasetClientPtr = weakPtr.lock()) {
      auto dataset = datasetClientPtr->getDataset();
      BufferAdaptor::Access buf(target.get());

      if (buf.numFrames() != dataset.size())
        return {Result::Status::kError, "Buffer and dataset do not match"};
      FluidTensor<double, 1> target(dataset.size());
      target = buf.samps(0, dataset.size(), 0);
      //FluidDataset(FluidTensor<idType, 1> ids, FluidTensor<dataType, N + 1> points, FluidTensor<targetType, 1> targets) {
      FluidTensor<string, 1> ids;
      FluidTensor<double, 2> data;
      ids = dataset.getIds();
      data = dataset.getData();
      auto regressionDataset = FluidDataset<string, double, double, 1>(ids, data, target);
      mTree = algorithm::KDTree<double>{regressionDataset};
    } else {
      return {Result::Status::kError, "Dataset doesn't exist"};
    }
    return {};
  }

  MessageResult<double> regress(BufferPtr data, int k) const {
    algorithm::KNNRegressor regressor;
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
      double result = regressor.predict(mTree, point, k);
      return result;
    }
  }

  FLUID_DECLARE_MESSAGES(makeMessage("index", &KNNRegClient::index),
                         makeMessage("regress", &KNNRegClient::regress));

private:
  mutable algorithm::KDTree<double> mTree{0};
};

using NRTThreadedKNNRegClient = NRTThreadingAdaptor<ClientWrapper<KNNRegClient>>;

} // namespace client
} // namespace fluid
