#pragma once

#include "NRTClient.hpp"
#include "algorithms/KNNRegressor.hpp"

namespace fluid {
namespace client {

  struct KNNRegressorData{
    algorithm::KDTree tree{0};
    FluidDataSet<std::string, double, 1> target{1};
    index size(){return target.size();}
    index dims(){return tree.dims();}
  };

  void to_json(nlohmann::json& j, const KNNRegressorData& data) {
    j["tree"] = data.tree;
    j["target"] = data.target;
  }
  void from_json(const nlohmann::json& j, KNNRegressorData& data) {
    data.tree = j["tree"].get<algorithm::KDTree>();
    data.target = j["target"].get<FluidDataSet<std::string, double, 1>>();
  }

class KNNRegressorClient : public FluidBaseClient, OfflineIn, OfflineOut, ModelObject {
  enum { kNDims, kK };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using DataSet = FluidDataSet<string, double, 1>;
  using StringVector = FluidTensor<string, 1>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS();

  KNNRegressorClient(ParamSetViewType &p) : mParams(p), mDataClient(mData) {}

  MessageResult<string> fit(
    DataSetClientRef datasetClient,
    DataSetClientRef targetClient)
    {
    auto datasetClientPtr = datasetClient.get().lock();
    if(!datasetClientPtr) return Error<string>(NoDataSet);
    auto dataSet = datasetClientPtr->getDataSet();
    if (dataSet.size() == 0) return Error<string>(EmptyDataSet);
    auto targetClientPtr = targetClient.get().lock();
    if(!targetClientPtr) return Error<string>(NoDataSet);
    auto target = targetClientPtr->getDataSet();
    if (target.size() == 0) return Error<string>(EmptyDataSet);
    if(dataSet.size() != target.size())
      return Error<string>("Different sizes for source and target");
    mTree = algorithm::KDTree{dataSet};
    mTarget = target;
    return {};
  }

  MessageResult<double> predictPoint(
    BufferPtr data, fluid::index k, bool uniform) const
    {
    algorithm::KNNRegressor regressor;
    if (!data) return Error<double>(NoBuffer);
    if(k == 0) return Error<double>(SmallK);
    if(mTree.size() == 0) return Error<double>(NoDataFitted);
    if (mTree.size() < k) return Error<double>(NotEnoughData);
    BufferAdaptor::Access buf(data.get());
    if(!buf.exists()) return Error<double>(InvalidBuffer);
    if (buf.numFrames() != mTree.dims()) return Error<double>(WrongPointSize);
    RealVector point(mTree.dims());
    point = buf.samps(0, mTree.dims(), 0);
    double result = regressor.predict(mTree, mTarget, point, k, !uniform);
    return result;
  }

  MessageResult<void> predict(
    DataSetClientRef source,
    DataSetClientRef dest, fluid::index k, bool uniform) const
    {
    auto sourcePtr = source.get().lock();
    if(!sourcePtr) return Error(NoDataSet);
    auto dataSet = sourcePtr->getDataSet();
    if (dataSet.size() == 0) return Error(EmptyDataSet);
    auto destPtr = dest.get().lock();
    if(!destPtr) return Error(NoDataSet);
    if (dataSet.pointSize()!=mTree.dims()) return Error(WrongPointSize);
    if(k == 0) return Error(SmallK);
    if(mTree.size() == 0) return Error(NoDataFitted);
    if (mTree.size() < k) return Error(NotEnoughData);

    algorithm::KNNRegressor regressor;
    auto ids = dataSet.getIds();
    auto data = dataSet.getData();
    DataSet result(1);
    for (index i = 0; i < dataSet.size(); i++) {
      RealVectorView point = data.row(i);
      RealVector prediction = {
        regressor.predict(mTree, mTarget, point, k, !uniform)
      };
      result.add(ids(i), prediction);
    }
    destPtr->setDataSet(result);
    return OK();
  }
  MessageResult<index> size() { return mDataClient.size(); }
  MessageResult<index> cols() { return mDataClient.dims(); }
  MessageResult<void> write(string fn) {return mDataClient.write(fn);}
  MessageResult<void> read(string fn) {return mDataClient.read(fn);}
  MessageResult<string> dump() { return mDataClient.dump();}
  MessageResult<void> load(string  s) { return mDataClient.load(s);}


  FLUID_DECLARE_MESSAGES(makeMessage("fit",
                         &KNNRegressorClient::fit),
                         makeMessage("predict",
                         &KNNRegressorClient::predict),
                         makeMessage("predictPoint",
                         &KNNRegressorClient::predictPoint),
                         makeMessage("cols",
                         &KNNRegressorClient::cols),
                         makeMessage("size",
                         &KNNRegressorClient::size),
                         makeMessage("load",
                         &KNNRegressorClient::load),
                         makeMessage("dump",
                         &KNNRegressorClient::dump),
                         makeMessage("write",
                         &KNNRegressorClient::write),
                         makeMessage("read",
                         &KNNRegressorClient::read)
                       );
private:
  algorithm::KDTree mTree{0};
  DataSet mTarget{1};
  KNNRegressorData mData{mTree,mTarget};
  DataClient<KNNRegressorData> mDataClient;
};

using NRTThreadedKNNRegressorClient = NRTThreadingAdaptor<ClientWrapper<KNNRegressorClient>>;

} // namespace client
} // namespace fluid
