#pragma once

//#include "DataSetClient.hpp"
#include "DataSetErrorStrings.hpp"
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

class KNNRegClient : public FluidBaseClient, OfflineIn, OfflineOut {
  enum { kNDims, kK };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS();

  KNNRegClient(ParamSetViewType &p) : mParams(p) {}

  MessageResult<std::string> index(DataSetClientRef sourceClient, DataSetClientRef targetClient) const {
    auto srcPtr = sourceClient.get().lock();
    auto tgtPtr = targetClient.get().lock();
    if (srcPtr && tgtPtr) {
      auto srcDataSet = srcPtr->getDataSet();
      mTarget = tgtPtr->getDataSet();
      std::cout<<srcDataSet.size()<<" "<<mTarget.size()<<std::endl;
      if (srcDataSet.size() != mTarget.size())
        return {Result::Status::kError, "Source and target size do not match"};
      mTree = algorithm::KDTree{srcDataSet};
    } else {
      return {Result::Status::kError, "DataSet doesn't exist"};
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
        return {Result::Status::kError, WrongPointSizeError};
      FluidTensor<double, 1> point(mTree.nDims());
      point = buf.samps(0, mTree.nDims(), 0);
      double result = regressor.predict(mTree, mTarget, point, k);
      return result;
    }
  }

  FLUID_DECLARE_MESSAGES(makeMessage("index", &KNNRegClient::index),
                         makeMessage("fit", &KNNRegClient::index),
                         makeMessage("regress", &KNNRegClient::regress));

private:
  mutable algorithm::KDTree mTree{0};
  mutable FluidDataSet<string, double, 1> mTarget{1};

};

using NRTThreadedKNNRegClient = NRTThreadingAdaptor<ClientWrapper<KNNRegClient>>;

} // namespace client
} // namespace fluid
