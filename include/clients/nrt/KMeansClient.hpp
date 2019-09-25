#pragma once

//#include "DatasetClient.hpp"
#include "DatasetErrorStrings.hpp"
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
#include <nlohmann/json.hpp>
#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>
#include<data/FluidFile.hpp>


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
      if (dataset.size() == 0) return {Result::Status::kError, EmptyDatasetError};
      if (init == nullptr){
        mModel.train(dataset, maxIter);
      }
      else {
        BufferAdaptor::Access buf(init.get());
        if (buf.numFrames() != mDims)
          return {Result::Status::kError,WrongPointSizeError};
        if(buf.numChans() != mK){
          return {Result::Status::kError,WrongInitError};
        }
        return {Result::Status::kError,"Not implemented"};

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
    return {};
  }

  MessageResult<int> predict(BufferPtr data) const {
    if (!data)
      return {Result::Status::kError, NoBufferError};
    BufferAdaptor::Access buf(data.get());
    if (buf.numFrames() != mDims)
      return {Result::Status::kError, WrongPointSizeError};

    FluidTensor<double, 1> point(mDims);
    point = buf.samps(0, mDims, 0);
    return mModel.vq(point);
  }

  MessageResult<void> write(string fileName) {
    auto file = FluidFile(fileName, "w");
    if(!file.valid()){return {Result::Status::kError, file.error()};}
    RealMatrix means(mK,mDims);
    mModel.getMeans(means);
    file.add("means", means);
    file.add("cols", mDims);
    file.add("rows", mK);
    return file.write()? mOKResult:mWriteError;
  }

  MessageResult<void> read(string fileName) {
    auto file = FluidFile(fileName, "r");
    if(!file.valid()){return {Result::Status::kError, file.error()};}
    if(!file.read()){return {Result::Status::kError, ReadError};}
    if(!file.checkKeys({"means","rows","cols"})){
      return {Result::Status::kError, file.error()};
    }
    file.get("cols", mDims);
    file.get("rows", mK);
    RealMatrix means(mK,mDims);
    file.get("means", means, mK, mDims);
    mModel  = algorithm::KMeans();
    mModel.init(mK, mDims);
    mModel.setMeans(means);
    return mOKResult;
  }

  FLUID_DECLARE_MESSAGES(makeMessage("train", &KMeansClient::train),
                         makeMessage("predict", &KMeansClient::predict),
                         makeMessage("write", &KMeansClient::write),
                         makeMessage("read", &KMeansClient::read));

private:
  MessageResult<void> mOKResult{Result::Status::kOk};
  MessageResult<void> mWriteError{Result::Status::kError, WriteError};
  mutable FluidDataset<string, double, string, 1> mDataset;
  mutable algorithm::KMeans mModel;
  size_t mDims;
  size_t mK;
};

using NRTThreadedKMeansClient = NRTThreadingAdaptor<ClientWrapper<KMeansClient>>;

} // namespace client
} // namespace fluid
