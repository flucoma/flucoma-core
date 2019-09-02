#pragma once

#include "DatasetClient.hpp"

#include "data/FluidDataset.hpp"
#include "algorithms/KMeans.hpp"

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

class KMeansClient : public FluidBaseClient, OfflineIn, OfflineOut {
  enum { kNDims, kK };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS(
    LongParam<Fixed<true>>("nDims", "Dimension size", 1,
                                              Min(1)),
    LongParam<Fixed<true>>("k", "Number of clusters", 2, Min(2)));

  KMeansClient(ParamSetViewType &p) : mParams(p), mDims(get<kNDims>()), mK(get<kK>()), mDataset(get<kNDims>()) {
    //mDims = get<kNDims>();
    mModel.init(mK, mDims);
  }

  MessageResult<void> train(DatasetClientRef datasetClient, int maxIter = 100, BufferPtr init = nullptr) {
    auto weakPtr = datasetClient.get();
    if(auto datasetClientPtr = weakPtr.lock())
    {
      auto dataset = datasetClientPtr->getDataset();
      dataset.print();
      if (init == nullptr){
        mModel.train(dataset, maxIter);
      }
      else {
        BufferAdaptor::Access buf(init.get());
        if (buf.numFrames() != mDims)
          return mWrongSizeError;
        if(buf.numChans() != mK){
          return mWrongInitError;
        }
        return {Result::Status::kError,"Not implememented"};
        FluidTensor<double, 2> points(mDims, mK);
        points = buf.samps(0, mDims, 0);
        mModel.train(dataset, maxIter, points);
      }
      //(const FluidDataset<std::string, double, std::string, 1> &dataset, int maxIter,
      //           RealMatrixView initialMeans = RealMatrixView(nullptr, 0, 0, 0))
    }
    else {
      return {Result::Status::kError,"Dataset doesn't exist"};
    }
    return mOKResult;
  }

  MessageResult<int> predict(BufferPtr data) const {
    if (!data)
      return mNoBufferError;
    BufferAdaptor::Access buf(data.get());
    if (buf.numFrames() != mDims)
      return mWrongSizeError;

    FluidTensor<double, 1> point(mDims);
    point = buf.samps(0, mDims, 0);
    return mModel.vq(point);
  }

  FLUID_DECLARE_MESSAGES(makeMessage("train", &KMeansClient::train),
                         makeMessage("predict", &KMeansClient::predict));

private:
  MessageResult<void> mNoBufferError{Result::Status::kError,
                                     "No buffer passed"};
  MessageResult<void> mNotFoundError{Result::Status::kError, "Point not found"};
  MessageResult<void> mWrongSizeError{Result::Status::kError,
                                      "Wrong point size"};
  MessageResult<void> mWrongInitError{Result::Status::kError,
                                      "Wrong number of initial points"};
  MessageResult<void> mDuplicateError{Result::Status::kError,
                                      "Label already in dataset"};
  MessageResult<void> mOKResult{Result::Status::kOk};
  mutable FluidDataset<string, double, string, 1> mDataset;
  mutable algorithm::KMeans mModel;
  size_t mDims;
  size_t mK;
};

using NRTThreadedKMeansClient = NRTThreadingAdaptor<ClientWrapper<KMeansClient>>;

} // namespace client
} // namespace fluid
