#pragma once

//#include "DataSetClient.hpp"
#include "DataSetErrorStrings.hpp"
#include "data/FluidDataSet.hpp"
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


public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS();

  KMeansClient(ParamSetViewType &p) : mParams(p) {
    //mDims = get<kNDims>();
    //mModel.init(mK, mDims);
  }

  MessageResult<FluidTensor<intptr_t, 1>> train(DataSetClientRef datasetClient, int k, int maxIter = 100, BufferPtr init = nullptr) {
    auto weakPtr = datasetClient.get();
    FluidTensor<intptr_t, 1> counts(k);
    counts.fill(0);
    if(auto datasetClientPtr = weakPtr.lock())
    {
      auto dataSet = datasetClientPtr->getDataSet();
      if (dataSet.size() == 0) return {Result::Status::kError, EmptyDataSetError};
      mDims = dataSet.pointSize();
      mK = k;
      mModel.init(mK, mDims);

      if (init == nullptr){
        mModel.train(dataSet, maxIter);
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
        mModel.train(dataSet, maxIter, points);
      }
      //(const FluidDataSet<std::string, double, std::string, 1> &dataset, int maxIter,
      //           RealMatrixView initialMeans = RealMatrixView(nullptr, 0, 0, 0))
      FluidTensor<int, 1> assignments(dataSet.size());
      mModel.getAssignments(assignments);
      for(auto a : assignments){
        counts[a]++;
      }
    }
    else {
      return {Result::Status::kError,"DataSet doesn't exist"};
    }
    return counts;
  }

  MessageResult<void> getClusters(DataSetClientRef datasetClient, LabelSetClientRef labelClient) {
    auto dataPtr = datasetClient.get().lock();
    auto labelPtr = labelClient.get().lock();
    if(!mModel.trained()){
      return {Result::Status::kError, "No data fitted"};
    }
    //FluidTensor<intptr_t, 1> counts(mModel.getK());
    //counts.fill(0);
    if(dataPtr && labelPtr)
    {
      auto dataSet = dataPtr->getDataSet();
      if (dataSet.size() == 0) return {Result::Status::kError, EmptyDataSetError};
      if (dataSet.size() != mModel.nAssigned()) return {Result::Status::kError, "Wrong number of points"};
      auto ids = dataSet.getIds();
      FluidTensor<int, 1> assignments(dataSet.size());
      mModel.getAssignments(assignments);
      FluidDataSet<string, string, 1> result(1);
      for(int i = 0; i < ids.size(); i++){
        int clusterId = assignments(i);
        //counts(clusterId)++;
        FluidTensor<string, 1> point = {std::to_string(clusterId)};
        result.add(ids(i), point);
      }
      labelPtr->setLabelSet(result);
    }
    else {
      return {Result::Status::kError,"Missing DataSet or LabelSet"};
    }
    //return counts;
    return {};
  }

  MessageResult<int> predictPoint(BufferPtr data) const {
    if (!data)
      return {Result::Status::kError, NoBufferError};
    BufferAdaptor::Access buf(data.get());
    if(!mModel.trained()){
      return {Result::Status::kError, "No data fitted"};
    }
    if (buf.numFrames() != mDims)
      return {Result::Status::kError, WrongPointSizeError};

    FluidTensor<double, 1> point(mDims);
    point = buf.samps(0, mDims, 0);
    return mModel.vq(point);
  }

  MessageResult<FluidTensor<intptr_t, 1>> predict(DataSetClientRef datasetClient, LabelSetClientRef labelClient) const {
    auto dataPtr = datasetClient.get().lock();
    auto labelPtr = labelClient.get().lock();
    if(!mModel.trained()){
      return {Result::Status::kError, "No data fitted"};
    }
    FluidTensor<intptr_t, 1> counts(mModel.getK());
    counts.fill(0);
    if(dataPtr && labelPtr)
    {
      auto dataSet = dataPtr->getDataSet();
      if (dataSet.size() == 0) return {Result::Status::kError, EmptyDataSetError};
      auto ids = dataSet.getIds();
      FluidTensor<double, 1> query(mDims);
      FluidDataSet<string, string, 1> result(1);

      for(int i = 0; i < ids.size(); i++){
        dataSet.get(ids(i), query);
        int clusterId = mModel.vq(query);
        counts(clusterId)++;
        FluidTensor<string, 1> point = {std::to_string(clusterId)};
        result.add(ids(i), point);
      }
      labelPtr->setLabelSet(result);
    }
    else {
      return {Result::Status::kError,"Missing DataSet or LabelSet"};
    }
    return counts;
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

  MessageResult<int> cols() { return mDims; }


  FLUID_DECLARE_MESSAGES(makeMessage("train", &KMeansClient::train),
                         makeMessage("fit", &KMeansClient::train),
                         makeMessage("predict", &KMeansClient::predict),
                         makeMessage("predictPoint", &KMeansClient::predictPoint),
                         makeMessage("getClusters", &KMeansClient::getClusters),
                         makeMessage("cols", &KMeansClient::cols),
                         makeMessage("write", &KMeansClient::write),
                         makeMessage("read", &KMeansClient::read));

private:
  MessageResult<void> mOKResult{Result::Status::kOk};
  MessageResult<void> mWriteError{Result::Status::kError, WriteError};
  mutable algorithm::KMeans mModel;
  size_t mDims;
  size_t mK;
};

using NRTThreadedKMeansClient = NRTThreadingAdaptor<ClientWrapper<KMeansClient>>;

} // namespace client
} // namespace fluid
