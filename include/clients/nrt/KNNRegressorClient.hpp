#pragma once

#include "NRTClient.hpp"
#include "algorithms/KNNRegressor.hpp"

namespace fluid {
namespace client {

class KNNRegressorClient : public FluidBaseClient, OfflineIn, OfflineOut, ModelObject {
  enum { kNDims, kK };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using DataSet = FluidDataSet<string, double, 1>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS();

  KNNRegressorClient(ParamSetViewType &p) : mParams(p) {}

  MessageResult<std::string> fit(
    DataSetClientRef datasetClient,
    DataSetClientRef targetClient)
    {
    auto datasetClientPtr = datasetClient.get().lock();
    if(!datasetClientPtr) return NoDataSetError;
    auto dataSet = datasetClientPtr->getDataSet();
    if (dataSet.size() == 0) return EmptyDataSetError;
    auto targetClientPtr = targetClient.get().lock();
    if(!targetClientPtr) return NoDataSetError;
    auto target = targetClientPtr->getDataSet();
    if (target.size() == 0) return EmptyDataSetError;
    if(dataSet.size() != target.size())
      return {Result::Status::kError, "Different sizes for source and target"};
    mTree = algorithm::KDTree{dataSet};
    mTarget = target;
    return OKResult;
  }

  MessageResult<double> predictPoint(
    BufferPtr data, fluid::index k) const
    {
    algorithm::KNNRegressor regressor;
    if (!data) return NoBufferError;
    if(k == 0) return SmallKError;
    if(mTree.nPoints() == 0) return NoDataFittedError;
    if (mTree.nPoints() < k) return NotEnoughDataError;
    BufferAdaptor::Access buf(data.get());
    if (buf.numFrames() != mTree.nDims()) return WrongPointSizeError;
    FluidTensor<double, 1> point(mTree.nDims());
    point = buf.samps(0, mTree.nDims(), 0);
    double result = regressor.predict(mTree, mTarget, point, k);
    return result;
  }

  MessageResult<void> predict(
    DataSetClientRef source,
    DataSetClientRef dest, fluid::index k) const
    {
    auto sourcePtr = source.get().lock();
    if(!sourcePtr) return NoDataSetError;
    auto dataSet = sourcePtr->getDataSet();
    if (dataSet.size() == 0) return EmptyDataSetError;
    auto destPtr = dest.get().lock();
    if(!destPtr) return NoDataSetError;
    if (dataSet.pointSize()!=mTree.nDims()) return WrongPointSizeError;
    if(k == 0) return SmallKError;
    if(mTree.nPoints() == 0) return NoDataFittedError;
    if (mTree.nPoints() < k) return NotEnoughDataError;

    algorithm::KNNRegressor regressor;
    auto ids = dataSet.getIds();
    auto data = dataSet.getData();
    DataSet result(1);
    for (index i = 0; i < dataSet.size(); i++) {
      FluidTensorView<double, 1> point = data.row(i);
      FluidTensor<double, 1> prediction = {
        regressor.predict(mTree, mTarget, point, k)
      };
      result.add(ids(i), prediction);
    }
    destPtr->setDataSet(result);
    return OKResult;
  }

  FLUID_DECLARE_MESSAGES(makeMessage("fit",
                         &KNNRegressorClient::fit),
                         makeMessage("predict",
                         &KNNRegressorClient::predict),
                         makeMessage("predictPoint", 
                         &KNNRegressorClient::predictPoint)
                       );
private:
  algorithm::KDTree mTree{0};
  DataSet mTarget{1};
};

using NRTThreadedKNNRegressorClient = NRTThreadingAdaptor<ClientWrapper<KNNRegressorClient>>;

} // namespace client
} // namespace fluid
