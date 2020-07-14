#pragma once

#include "NRTClient.hpp"
#include "DataSetClient.hpp"
#include "LabelSetClient.hpp"
#include "algorithms/LabelSetEncoder.hpp"
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
  return fluid::check_json(j,
    {"tree", "labels"}, {JSONTypes::OBJECT, JSONTypes::OBJECT}
  );
}

void from_json(const nlohmann::json& j, KNNClassifierData& data) {
  data.tree = j.at("tree").get<algorithm::KDTree>();
  data.labels = j.at("labels").get<FluidDataSet<std::string, std::string, 1>>();
}

class KNNClassifierClient : public FluidBaseClient,
                     AudioIn,
                     ControlOut,
                     ModelObject,
                     public DataClient<KNNClassifierData> {

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using LabelSet = FluidDataSet<string, string, 1>;
  using DataSet = FluidDataSet<string, double, 1>;
  using StringVector = FluidTensor<string, 1>;

  enum {kNumNeighbors, kWeight, kInputBuffer, kOutputBuffer};

  FLUID_DECLARE_PARAMS(
    LongParam("numNeighbours","Number of Nearest Neighbours", 3, Min(1)),
    EnumParam("weight", "Weight Neighbours by Distance", 1, "No", "Yes"),
    BufferParam("inputPointBuffer","Input Point Buffer"),
    BufferParam("predictionBuffer","Prediction Buffer")
  );

  KNNClassifierClient(ParamSetViewType &p) : mParams(p)
  {
    audioChannelsIn(1);
    controlChannelsOut(1);
  }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>> &input,
               std::vector<FluidTensorView<T, 1>> &output, FluidContext &)
  {
    index k = get<kNumNeighbors>();
    bool weight = get<kWeight>() != 0;
    if(k == 0 || mTree.size() == 0 || mTree.size() < k) return;
    InOutBuffersCheck bufCheck(mTree.dims());
    if(!bufCheck.checkInputs(
      get<kInputBuffer>().get(),
      get<kOutputBuffer>().get()))
      return;
    algorithm::KNNClassifier classifier;
    RealVector point(mTree.dims());
    point = BufferAdaptor::ReadAccess(get<kInputBuffer>().get()).samps(0, mTree.dims(), 0);
    mTrigger.process(input, output, [&](){
      std::string result = classifier.predict(mTree, point, mLabels, k, weight);
      BufferAdaptor::Access(get<kOutputBuffer>().get()).samps(0)[0] = static_cast<double>(
        mLabelSetEncoder.encodeIndex(result)
      );
    });
  }

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
    if(dataset.size() != labelSet.size())return Error<string>(SizesDontMatch);
    mTree = algorithm::KDTree{dataset};
    mLabels = labelSet;
    mAlgorithm = {mTree, mLabels};
    mLabelSetEncoder.fit(mLabels);
    return {};
  }

  MessageResult<string> predictPoint(
    BufferPtr data) const
  {
    index k = get<kNumNeighbors>();
    bool weight = get<kWeight>() != 0;
    if(k == 0) return Error<string>(SmallK);
    if(mTree.size() == 0) return Error<string>(NoDataFitted);
    if (mTree.size() < k) return Error<string>(NotEnoughData);
    InBufferCheck bufCheck(mTree.dims());
    if(!bufCheck.checkInputs(data.get())) return Error<string>(bufCheck.error());
    algorithm::KNNClassifier classifier;
    RealVector point(mTree.dims());
    point = BufferAdaptor::ReadAccess(data.get()).samps(0, mTree.dims(), 0);
    std::string result = classifier.predict(mTree, point, mLabels, k, weight);
    return result;
  }

  MessageResult<void> predict(
    DataSetClientRef source,
    LabelSetClientRef dest) const
  {
    index k = get<kNumNeighbors>();
    bool weight = get<kWeight>() != 0;
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
        classifier.predict(mTree, point, mLabels, k, weight)
      };
      result.add(ids(i), label);
    }
    destPtr->setLabelSet(result);
    return OK();
  }
  index latency() { return 0; }

  FLUID_DECLARE_MESSAGES(makeMessage("fit",
                         &KNNClassifierClient::fit),
                         makeMessage("predict",
                         &KNNClassifierClient::predict),
                         makeMessage("predictPoint",
                         &KNNClassifierClient::predictPoint),
                         makeMessage("cols", &KNNClassifierClient::dims),
                         makeMessage("size", &KNNClassifierClient::size),
                         makeMessage("load", &KNNClassifierClient::load),
                         makeMessage("dump", &KNNClassifierClient::dump),
                         makeMessage("write", &KNNClassifierClient::write),
                         makeMessage("read", &KNNClassifierClient::read));

private:
  algorithm::KDTree mTree{0};
  LabelSet mLabels{1};
  FluidInputTrigger mTrigger;
  algorithm::LabelSetEncoder mLabelSetEncoder;
};

using RTKNNClassifierClient = ClientWrapper<KNNClassifierClient>;
} // namespace client
} // namespace fluid
