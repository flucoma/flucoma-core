#pragma once

#include "NRTClient.hpp"
#include "algorithms/KNNClassifier.hpp"

namespace fluid {
namespace client {

struct KNNClassifierData{
  algorithm::KDTree tree{0};
  FluidDataSet<std::string, std::string, 1> labels{1};
  index size(){return labels.size();}
  index dims(){return tree.dims();}
};

void to_json(nlohmann::json& j, const KNNClassifierData& data) {
  j["tree"] = data.tree;
  j["labels"] = data.labels;
}

bool check_json(const nlohmann::json& j, const KNNClassifierData&){
  return fluid::check_json(j, {"tree", "labels"});
}

void from_json(const nlohmann::json& j, KNNClassifierData& data) {
  data.tree = j.at("tree").get<algorithm::KDTree>();
  data.labels = j.at("labels").get<FluidDataSet<std::string, std::string, 1>>();
}

class KNNClassifierClient : public FluidBaseClient, OfflineIn, OfflineOut, ModelObject {
  enum { kNDims, kK };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using LabelSet = FluidDataSet<string, string, 1>;
  using DataSet = FluidDataSet<string, double, 1>;
  using StringVector = FluidTensor<string, 1>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS();

  KNNClassifierClient(ParamSetViewType &p) : mParams(p), mDataClient(mData) {}

  MessageResult<string> fit(
    DataSetClientRef datasetClient,
    LabelSetClientRef labelsetClient)
    {
    auto datasetClientPtr = datasetClient.get().lock();
    if(!datasetClientPtr) return Error<string>(NoDataSet);
    auto dataset = datasetClientPtr->getDataSet();
    if (dataset.size() == 0) return Error<string>(EmptyDataSet);
    auto labelsetPtr = labelsetClient.get().lock();
    if(!labelsetPtr) return Error<string>(NoLabelSet);
    auto labelSet = labelsetPtr->getLabelSet();
    if (labelSet.size() == 0) return Error<string>(EmptyLabelSet);
    if(dataset.size() != labelSet.size())
      return Error<string>("Different sizes for source and target");
    mTree = algorithm::KDTree{dataset};
    mLabels = labelSet;
    mData = {mTree,mLabels};
    return {};
  }

  MessageResult<string> predictPoint(
    BufferPtr data, fluid::index k, bool uniform) const
    {
    algorithm::KNNClassifier classifier;
    if (!data) return Error<string>(NoBuffer);
    if(k == 0) return Error<string>(SmallK);
    if(mTree.size() == 0) return Error<string>(NoDataFitted);
    if (mTree.size() < k) return Error<string>(NotEnoughData);
    BufferAdaptor::Access buf(data.get());
    if(!buf.exists()) return Error<string>(InvalidBuffer);
    if (buf.numFrames() != mTree.dims()) return Error<string>(WrongPointSize);

    RealVector point(mTree.dims());
    point = buf.samps(0, mTree.dims(), 0);
    std::string result = classifier.predict(mTree, point, mLabels, k, !uniform);
    return result;
  }

  MessageResult<void> predict(
    DataSetClientRef source,
    LabelSetClientRef dest, fluid::index k, bool uniform) const
    {
    auto sourcePtr = source.get().lock();
    if(!sourcePtr) return Error(NoDataSet);
    auto dataSet = sourcePtr->getDataSet();
    if (dataSet.size() == 0) return Error(EmptyDataSet);
    auto destPtr = dest.get().lock();
    if(!destPtr) return Error(NoLabelSet);
    if (dataSet.pointSize()!=mTree.dims()) return Error(WrongPointSize);
    if(k == 0) return Error(SmallK);
    if(mTree.size() == 0) return Error(NoDataFitted);
    if (mTree.size() < k) return Error(NotEnoughData);

    algorithm::KNNClassifier classifier;
    auto ids = dataSet.getIds();
    auto data = dataSet.getData();
    LabelSet result(1);
    for (index i = 0; i < dataSet.size(); i++) {
      RealVectorView point = data.row(i);
      StringVector label = {
        classifier.predict(mTree, point, mLabels, k, !uniform)
      };
      result.add(ids(i), label);
    }
    destPtr->setLabelSet(result);
    return OK();
  }

  MessageResult<index> size() { return mDataClient.size(); }
  MessageResult<index> cols() { return mDataClient.dims(); }
  MessageResult<void> write(string fn) {return mDataClient.write(fn);}
  MessageResult<void> read(string fn) {return mDataClient.read(fn);}
  MessageResult<string> dump() { return mDataClient.dump();}
  MessageResult<void> load(string  s) { return mDataClient.load(s);}


  FLUID_DECLARE_MESSAGES(makeMessage("fit",
                         &KNNClassifierClient::fit),
                         makeMessage("predict",
                         &KNNClassifierClient::predict),
                         makeMessage("predictPoint",
                         &KNNClassifierClient::predictPoint),
                         makeMessage("cols", &KNNClassifierClient::cols),
                         makeMessage("size", &KNNClassifierClient::size),
                         makeMessage("load", &KNNClassifierClient::load),
                         makeMessage("dump", &KNNClassifierClient::dump),
                         makeMessage("write", &KNNClassifierClient::write),
                         makeMessage("read", &KNNClassifierClient::read));

private:
  algorithm::KDTree mTree{0};
  LabelSet mLabels{1};
  KNNClassifierData mData{mTree,mLabels};
  DataClient<KNNClassifierData> mDataClient;
};

using NRTThreadedKNNClassifierClient = NRTThreadingAdaptor<ClientWrapper<KNNClassifierClient>>;

} // namespace client
} // namespace fluid
