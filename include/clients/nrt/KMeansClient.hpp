#pragma once

#include "NRTClient.hpp"
#include "algorithms/KMeans.hpp"
#include <string>

namespace fluid {
namespace client {

class KMeansClient : public FluidBaseClient, OfflineIn, OfflineOut {

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS();

  KMeansClient(ParamSetViewType &p) : mParams(p) {
    // mDims = get<kNDims>();
    // mModel.init(mK, mDims);
  }

  MessageResult<FluidTensor<index, 1>> train(DataSetClientRef datasetClient,
                                             index k, index maxIter,
                                             BufferPtr init = nullptr) {
    auto weakPtr = datasetClient.get();
    FluidTensor<index, 1> counts(k);
    counts.fill(0);
    if (k <= 1)
      return SmallKError;
    if (maxIter <= 0)
      maxIter = 100;
    if (auto datasetClientPtr = weakPtr.lock()) {
      auto dataSet = datasetClientPtr->getDataSet();
      if (dataSet.size() == 0)
        return EmptyDataSetError;
      mDims = dataSet.pointSize();
      mK = k;
      mModel.init(mK, mDims);

      if (init == nullptr) {
        mModel.train(dataSet, maxIter);
      } else {
        BufferAdaptor::Access buf(init.get());
        if (buf.numFrames() != mDims)
          return WrongPointSizeError;
        if (buf.numChans() != mK) {
          return WrongInitError;
        }
        return NotImplementedError;

        FluidTensor<double, 2> points(mDims, mK);
        points = buf.samps(0, mDims, 0);
        mModel.train(dataSet, maxIter, points);
      }
      //(const FluidDataSet<std::string, double, std::string, 1> &dataset, int
      //maxIter,
      //           RealMatrixView initialMeans = RealMatrixView(nullptr, 0, 0,
      //           0))
      FluidTensor<index, 1> assignments(dataSet.size());
      mModel.getAssignments(assignments);
      for (auto a : assignments) {
        counts[a]++;
      }
    } else {
      return NoDataSetError;
    }
    return counts;
  }

  MessageResult<void> getClusters(DataSetClientRef datasetClient,
                                  LabelSetClientRef labelClient) {
    auto dataPtr = datasetClient.get().lock();
    auto labelPtr = labelClient.get().lock();
    if (!mModel.trained()) {
      return NoDataFittedError;
    }
    // FluidTensor<intptr_t, 1> counts(mModel.getK());
    // counts.fill(0);
    if (dataPtr && labelPtr) {
      auto dataSet = dataPtr->getDataSet();
      if (dataSet.size() == 0)
        return EmptyDataSetError;
      if (dataSet.size() != mModel.nAssigned())
        return WrongPointNumError;
      auto ids = dataSet.getIds();
      FluidTensor<index, 1> assignments(dataSet.size());
      mModel.getAssignments(assignments);
      FluidDataSet<string, string, 1> result(1);
      for (index i = 0; i < ids.size(); i++) {
        index clusterId = assignments(i);
        // counts(clusterId)++;
        FluidTensor<string, 1> point = {std::to_string(clusterId)};
        result.add(ids(i), point);
      }
      labelPtr->setLabelSet(result);
    } else {
      return NoDataSetOrLabelSetError;
    }
    // return counts;
    return {};
  }

  MessageResult<index> predictPoint(BufferPtr data) const {
    if (!data)
      return NoBufferError;
    BufferAdaptor::Access buf(data.get());
    if (!mModel.trained()) {
      return NoDataFittedError;
    }
    if (buf.numFrames() != mDims)
      return WrongPointSizeError;

    FluidTensor<double, 1> point(mDims);
    point = buf.samps(0, mDims, 0);
    return mModel.vq(point);
  }

  MessageResult<FluidTensor<index, 1>>
  predict(DataSetClientRef datasetClient, LabelSetClientRef labelClient) const {

    auto dataPtr = datasetClient.get().lock();
    if (!dataPtr) return NoDataSetError;
    auto labelPtr = labelClient.get().lock();
    if (!labelPtr) return NoLabelSetError;
    auto dataSet = dataPtr->getDataSet();
    if (dataSet.size() == 0) return EmptyDataSetError;
    if (!mModel.trained()) return NoDataFittedError;

    FluidTensor<index, 1> counts(mModel.getK());
    counts.fill(0);
    auto ids = dataSet.getIds();
    FluidTensor<double, 1> query(mDims);
    FluidDataSet<string, string, 1> result(1);
    for (index i = 0; i < ids.size(); i++) {
        dataSet.get(ids(i), query);
        index clusterId = mModel.vq(query);
        counts(clusterId)++;
        FluidTensor<string, 1> point = {std::to_string(clusterId)};
        result.add(ids(i), point);
    }
    labelPtr->setLabelSet(result);
    return counts;
  }

  MessageResult<void> write(string fileName) {
    auto file = FluidFile(fileName, "w");
    if (!file.valid()) {
      return {Result::Status::kError, file.error()};
    }
    RealMatrix means(mK, mDims);
    mModel.getMeans(means);
    file.add("means", means);
    file.add("cols", mDims);
    file.add("rows", mK);
    return file.write() ? OKResult : WriteError;
  }

  MessageResult<void> read(string fileName) {
    auto file = FluidFile(fileName, "r");
    if (!file.valid()) {
      return {Result::Status::kError, file.error()};
    }
    if (!file.read()) {
      return ReadError;
    }
    if (!file.checkKeys({"means", "rows", "cols"})) {
      return {Result::Status::kError, file.error()};
    }
    file.get("cols", mDims);
    file.get("rows", mK);
    RealMatrix means(mK, mDims);
    file.get("means", means, mK, mDims);
    mModel = algorithm::KMeans();
    mModel.init(mK, mDims);
    mModel.setMeans(means);
    return OKResult;
  }

  MessageResult<index> cols() { return mDims; }

  FLUID_DECLARE_MESSAGES(makeMessage("train", &KMeansClient::train),
                         makeMessage("fit", &KMeansClient::train),
                         makeMessage("predict", &KMeansClient::predict),
                         makeMessage("predictPoint",
                                     &KMeansClient::predictPoint),
                         makeMessage("getClusters", &KMeansClient::getClusters),
                         makeMessage("cols", &KMeansClient::cols),
                         makeMessage("write", &KMeansClient::write),
                         makeMessage("read", &KMeansClient::read));

private:
  algorithm::KMeans mModel;
  index mDims;
  index mK;
};

using NRTThreadedKMeansClient =
    NRTThreadingAdaptor<ClientWrapper<KMeansClient>>;

} // namespace client
} // namespace fluid
