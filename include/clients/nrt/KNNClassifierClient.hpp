/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "NRTClient.hpp"
#include "DataSetClient.hpp"
#include "LabelSetClient.hpp"
#include "../../algorithms/public/LabelSetEncoder.hpp"
#include "../../algorithms/public/KNNClassifier.hpp"

namespace fluid {
namespace client {
namespace knnclassifier{

struct KNNClassifierData{
  algorithm::KDTree tree{0};
  FluidDataSet<std::string, std::string, 1> labels{1};
  index size(){return labels.size();}
  index dims(){return tree.dims();}
  void clear(){labels = FluidDataSet<std::string, std::string, 1>(1);tree.clear();}
  bool initialized() const{return tree.initialized();}
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

enum { kNumNeighbors, kWeight, kInputBuffer, kOutputBuffer };

constexpr auto KNNClassifierParams = defineParameters(
    LongParam("numNeighbours", "Number of Nearest Neighbours", 3, Min(1)),
    EnumParam("weight", "Weight Neighbours by Distance", 1, "No", "Yes"),
    BufferParam("inputPointBuffer", "Input Point Buffer"),
    BufferParam("predictionBuffer", "Prediction Buffer"));

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

  using ParamDescType = decltype(KNNClassifierParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors()
  {
    return KNNClassifierParams;
  }

  KNNClassifierClient(ParamSetViewType &p) : mParams(p)
  {
    audioChannelsIn(1);
    controlChannelsOut({1,1});
  }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>> &input,
               std::vector<FluidTensorView<T, 1>> &output, FluidContext &)
  {
    index k = get<kNumNeighbors>();
    bool weight = get<kWeight>() != 0;
    if(k == 0 || mAlgorithm.tree.size() == 0 || mAlgorithm.tree.size() < k) return;
    InOutBuffersCheck bufCheck(mAlgorithm.tree.dims());
    if(!bufCheck.checkInputs(
      get<kInputBuffer>().get(),
      get<kOutputBuffer>().get()))
      return;
    auto outBuf = BufferAdaptor::Access(get<kOutputBuffer>().get());
    if(outBuf.samps(0).size() != 1) return;
    algorithm::KNNClassifier classifier;
    RealVector point(mAlgorithm.tree.dims());
    point = BufferAdaptor::ReadAccess(get<kInputBuffer>().get()).samps(0, mAlgorithm.tree.dims(), 0);
    mTrigger.process(input, output, [&](){
      std::string result = classifier.predict(mAlgorithm.tree, point, mAlgorithm.labels, k, weight);
      outBuf.samps(0)[0] = static_cast<double>(mLabelSetEncoder.encodeIndex(result));
    });
  }

  MessageResult<void> fit(
    DataSetClientRef datasetClient,
    LabelSetClientRef labelsetClient)
    {
    auto datasetClientPtr = datasetClient.get().lock();
    if(!datasetClientPtr) return Error(NoDataSet);
    auto dataset = datasetClientPtr->getDataSet();
    if (dataset.size() == 0) return Error(EmptyDataSet);
    auto labelsetPtr = labelsetClient.get().lock();
    if(!labelsetPtr) return Error(NoLabelSet);
    auto labelSet = labelsetPtr->getLabelSet();
    if (labelSet.size() == 0) return Error(EmptyLabelSet);
    if(dataset.size() != labelSet.size()) return Error(SizesDontMatch);
    mAlgorithm.tree = algorithm::KDTree{dataset};
    mAlgorithm.labels = labelSet;
    mAlgorithm = {mAlgorithm.tree, mAlgorithm.labels};
    mLabelSetEncoder.fit(mAlgorithm.labels);
    return OK();
  }

  MessageResult<string> predictPoint(
    BufferPtr data) const
  {
    index k = get<kNumNeighbors>();
    bool weight = get<kWeight>() != 0;
    if(k == 0) return Error<string>(SmallK);
    if(mAlgorithm.tree.size() == 0) return Error<string>(NoDataFitted);
    if (mAlgorithm.tree.size() < k) return Error<string>(NotEnoughData);
    InBufferCheck bufCheck(mAlgorithm.tree.dims());
    if(!bufCheck.checkInputs(data.get())) return Error<string>(bufCheck.error());
    algorithm::KNNClassifier classifier;
    RealVector point(mAlgorithm.tree.dims());
    point = BufferAdaptor::ReadAccess(data.get()).samps(0, mAlgorithm.tree.dims(), 0);
    std::string result = classifier.predict(mAlgorithm.tree, point, mAlgorithm.labels, k, weight);
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
    if (dataSet.pointSize() != mAlgorithm.tree.dims()) return Error(WrongPointSize);
    if(k == 0) return Error(SmallK);
    if(mAlgorithm.tree.size() == 0) return Error(NoDataFitted);
    if (mAlgorithm.tree.size() < k) return Error(NotEnoughData);

    algorithm::KNNClassifier classifier;
    auto ids = dataSet.getIds();
    auto data = dataSet.getData();
    LabelSet result(1);
    for (index i = 0; i < dataSet.size(); i++) {
      RealVectorView point = data.row(i);
      StringVector label = {
        classifier.predict(mAlgorithm.tree, point, mAlgorithm.labels, k, weight)
      };
      result.add(ids(i), label);
    }
    destPtr->setLabelSet(result);
    return OK();
  }
  index latency() { return 0; }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("fit", &KNNClassifierClient::fit),
        makeMessage("predict", &KNNClassifierClient::predict),
        makeMessage("predictPoint", &KNNClassifierClient::predictPoint),
        makeMessage("cols", &KNNClassifierClient::dims),
        makeMessage("clear", &KNNClassifierClient::clear),
        makeMessage("size", &KNNClassifierClient::size),
        makeMessage("load", &KNNClassifierClient::load),
        makeMessage("dump", &KNNClassifierClient::dump),
        makeMessage("write", &KNNClassifierClient::write),
        makeMessage("read", &KNNClassifierClient::read));
  }

private:
  FluidInputTrigger mTrigger;
  algorithm::LabelSetEncoder mLabelSetEncoder;
};
}

using RTKNNClassifierClient = ClientWrapper<knnclassifier::KNNClassifierClient>;
} // namespace client
} // namespace fluid
