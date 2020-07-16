#pragma once

#include "DataSetClient.hpp"
#include "NRTClient.hpp"
#include "algorithms/KNNRegressor.hpp"

namespace fluid {
namespace client {

struct KNNRegressorData {
  algorithm::KDTree tree{0};
  FluidDataSet<std::string, double, 1> target{1};
  index size() { return target.size(); }
  index dims() { return tree.dims(); }
};

void to_json(nlohmann::json &j, const KNNRegressorData &data) {
  j["tree"] = data.tree;
  j["target"] = data.target;
}

bool check_json(const nlohmann::json &j, const KNNRegressorData &) {
  return fluid::check_json(j, {"tree", "target"},
                           {JSONTypes::OBJECT, JSONTypes::OBJECT});
}

void from_json(const nlohmann::json &j, KNNRegressorData &data) {
  data.tree = j["tree"].get<algorithm::KDTree>();
  data.target = j["target"].get<FluidDataSet<std::string, double, 1>>();
}

class KNNRegressorClient : public FluidBaseClient,
                           AudioIn,
                           ControlOut,
                           ModelObject,
                           public DataClient<KNNRegressorData> {

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using DataSet = FluidDataSet<string, double, 1>;
  using StringVector = FluidTensor<string, 1>;

  enum {kNumNeighbors, kWeight, kInputBuffer, kOutputBuffer};

  FLUID_DECLARE_PARAMS(
    LongParam("numNeighbours","Number of Nearest Neighbours", 3, Min(1)),
    EnumParam("weight", "Weight Neighbours by Distance", 1, "No", "Yes"),
    BufferParam("inputPointBuffer","Input Point Buffer"),
    BufferParam("predictionBuffer","Prediction Buffer")
  );

  KNNRegressorClient(ParamSetViewType &p) : mParams(p) {
    audioChannelsIn(1);
    controlChannelsOut(1);
  }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>> &input,
               std::vector<FluidTensorView<T, 1>> &output, FluidContext &) {
    index k = get<kNumNeighbors>();
    bool weight = get<kWeight>() != 0;
    if (k == 0 || mAlgorithm.tree.size() == 0 || mAlgorithm.tree.size() < k)
      return;
    InOutBuffersCheck bufCheck(mAlgorithm.tree.dims());
    if (!bufCheck.checkInputs(get<kInputBuffer>().get(),
                              get<kOutputBuffer>().get()))
      return;
    algorithm::KNNRegressor regressor;
    RealVector point(mAlgorithm.tree.dims());
    point = BufferAdaptor::ReadAccess(get<kInputBuffer>().get()).samps(0, mAlgorithm.tree.dims(), 0);
    mTrigger.process(input, output, [&]() {
      double result = regressor.predict(mAlgorithm.tree, mAlgorithm.target, point, k, weight);
        BufferAdaptor::Access(get<kOutputBuffer>().get()).samps(0)[0] = result;
    });
  }

  index latency() { return 0; }

  MessageResult<string> fit(DataSetClientRef datasetClient,
                            DataSetClientRef targetClient) {
    auto datasetClientPtr = datasetClient.get().lock();
    if (!datasetClientPtr)
      return Error<string>(NoDataSet);
    auto dataSet = datasetClientPtr->getDataSet();
    if (dataSet.size() == 0)
      return Error<string>(EmptyDataSet);
    auto targetClientPtr = targetClient.get().lock();
    if (!targetClientPtr)
      return Error<string>(NoDataSet);
    auto target = targetClientPtr->getDataSet();
    if (target.size() == 0)
      return Error<string>(EmptyDataSet);
    if (dataSet.size() != target.size())
      return Error<string>(SizesDontMatch);
    mAlgorithm.tree = algorithm::KDTree{dataSet};
    mAlgorithm.target = target;
    mAlgorithm = {mAlgorithm.tree, mAlgorithm.target};
    return {};
  }

  MessageResult<double> predictPoint(BufferPtr data) const {
    index k = get<kNumNeighbors>();
    bool weight = get<kWeight>() != 0;
    if (k == 0) return Error<double>(SmallK);
    if (mAlgorithm.tree.size() == 0) return Error<double>(NoDataFitted);
    if (mAlgorithm.tree.size() < k) return Error<double>(NotEnoughData);
    InBufferCheck bufCheck(mAlgorithm.tree.dims());
    if (!bufCheck.checkInputs(data.get()))
      return Error<double>(bufCheck.error());
    algorithm::KNNRegressor regressor;
    RealVector point(mAlgorithm.tree.dims());
    point = BufferAdaptor::ReadAccess(data.get()).samps(0, mAlgorithm.tree.dims(), 0);
    double result = regressor.predict(mAlgorithm.tree, mAlgorithm.target, point, k, weight);
    return result;
  }

  MessageResult<void> predict(DataSetClientRef source, DataSetClientRef dest) const {
    index k = get<kNumNeighbors>();
    bool weight = get<kWeight>() != 0;
    auto sourcePtr = source.get().lock();
    if (!sourcePtr) return Error(NoDataSet);
    auto dataSet = sourcePtr->getDataSet();
    if (dataSet.size() == 0) return Error(EmptyDataSet);
    auto destPtr = dest.get().lock();
    if (!destPtr) return Error(NoDataSet);
    if (dataSet.pointSize() != mAlgorithm.tree.dims()) return Error(WrongPointSize);
    if (k == 0) return Error(SmallK);
    if (mAlgorithm.tree.size() == 0) return Error(NoDataFitted);
    if (mAlgorithm.tree.size() < k) return Error(NotEnoughData);

    algorithm::KNNRegressor regressor;
    auto ids = dataSet.getIds();
    auto data = dataSet.getData();
    DataSet result(1);
    for (index i = 0; i < dataSet.size(); i++) {
      RealVectorView point = data.row(i);
      RealVector prediction = {
        regressor.predict(mAlgorithm.tree, mAlgorithm.target, point, k, weight)
      };
      result.add(ids(i), prediction);
    }
    destPtr->setDataSet(result);
    return OK();
  }

  FLUID_DECLARE_MESSAGES(makeMessage("fit", &KNNRegressorClient::fit),
                         makeMessage("predict", &KNNRegressorClient::predict),
                         makeMessage("predictPoint",
                                     &KNNRegressorClient::predictPoint),
                         makeMessage("cols", &KNNRegressorClient::dims),
                         makeMessage("size", &KNNRegressorClient::size),
                         makeMessage("load", &KNNRegressorClient::load),
                         makeMessage("dump", &KNNRegressorClient::dump),
                         makeMessage("write", &KNNRegressorClient::write),
                         makeMessage("read", &KNNRegressorClient::read));

private:
  FluidInputTrigger mTrigger;
};

using RTKNNRegressorClient = ClientWrapper<KNNRegressorClient>;

} // namespace client
} // namespace fluid
