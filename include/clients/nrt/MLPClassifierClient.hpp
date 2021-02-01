#pragma once

#include "DataSetClient.hpp"
#include "LabelSetClient.hpp"
#include "NRTClient.hpp"
#include "algorithms/LabelSetEncoder.hpp"
#include "algorithms/MLP.hpp"
#include "algorithms/SGD.hpp"
#include <string>

namespace fluid {
namespace client {

struct MLPClassifierData {
  algorithm::MLP mlp;
  algorithm::LabelSetEncoder encoder;
  index size() { return mlp.size(); }
  index dims() { return mlp.dims(); }
  void clear() {
    mlp.clear();
    encoder.clear();
  }
  bool initialized() const {return mlp.initialized();}
};

void to_json(nlohmann::json &j, const MLPClassifierData &data) {
  j["mlp"] = data.mlp;
  j["labels"] = data.encoder;
}

bool check_json(const nlohmann::json &j, const MLPClassifierData &) {
  return fluid::check_json(j, {"mlp", "labels"},
                           {JSONTypes::OBJECT, JSONTypes::OBJECT});
}

void from_json(const nlohmann::json &j, MLPClassifierData &data) {
  data.mlp = j.at("mlp").get<algorithm::MLP>();
  data.encoder = j.at("labels").get<algorithm::LabelSetEncoder>();
}

class MLPClassifierClient : public FluidBaseClient,
                            AudioIn,
                            ControlOut,
                            ModelObject,
                            public DataClient<MLPClassifierData> {
  enum {
    kHidden,
    kActivation,
    kIter,
    kRate,
    kMomentum,
    kBatchSize,
    kVal,
    kInputBuffer,
    kOutputBuffer
  };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using IndexVector = FluidTensor<index, 1>;
  using StringVector = FluidTensor<string, 1>;
  using DataSet = FluidDataSet<string, double, 1>;
  using LabelSet = FluidDataSet<string, string, 1>;


  static constexpr std::initializer_list<index> HiddenLayerDefaults = {3, 3};

  FLUID_DECLARE_PARAMS(
      LongArrayParam("hidden", "Hidden Layer Sizes", HiddenLayerDefaults),
      EnumParam("activation", "Activation Function", 2, "Identity", "Sigmoid",
                "ReLU", "Tanh"),
      LongParam("maxIter", "Maximum Number of Iterations", 1000,  Min(1)),
      FloatParam("learnRate", "Learning Rate", 0.01, Min(0.0), Max(1.0)),
      FloatParam("momentum", "Momentum", 0.5, Min(0.0), Max(0.99)),
      LongParam("batchSize", "Batch Size", 50),
      FloatParam("validation", "Validation Amount", 0.2, Min(0), Max(0.9)),
      BufferParam("inputPointBuffer", "Input Point Buffer"),
      BufferParam("predictionBuffer", "Prediction Buffer")
);

  MLPClassifierClient(ParamSetViewType &p) : mParams(p) {
    audioChannelsIn(1);
    controlChannelsOut(1);
  }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>> &input,
               std::vector<FluidTensorView<T, 1>> &output, FluidContext &) {
    if (!mAlgorithm.mlp.trained())
      return;
    index dims = mAlgorithm.mlp.dims();
    index layer = mAlgorithm.mlp.size();

    InOutBuffersCheck bufCheck(dims);
    if (!bufCheck.checkInputs(get<kInputBuffer>().get(),
                              get<kOutputBuffer>().get()))
      return;
    auto outBuf = BufferAdaptor::Access(get<kOutputBuffer>().get());
    if(outBuf.samps(0).size() != 1) return;

    RealVector src(dims);
    RealVector dest(mAlgorithm.mlp.outputSize(layer));
    src =
        BufferAdaptor::ReadAccess(get<kInputBuffer>().get()).samps(0, dims, 0);
    mTrigger.process(input, output, [&]() {
      mAlgorithm.mlp.processFrame(src, dest, 0, layer);
      auto label = mAlgorithm.encoder.decodeOneHot(dest);
       outBuf.samps(0)[0] = static_cast<double>(
        mAlgorithm.encoder.encodeIndex(label)
      );
    });
  }

  index latency() { return 0; }

  MessageResult<double> fit(DataSetClientRef source, LabelSetClientRef target) {
    auto sourceClientPtr = source.get().lock();
    if (!sourceClientPtr)
      return Error<double>(NoDataSet);
    auto sourceDataSet = sourceClientPtr->getDataSet();
    if (sourceDataSet.size() == 0)
      return Error<double>(EmptyDataSet);
    if(mAlgorithm.initialized() && sourceDataSet.dims() != mAlgorithm.dims())
      return Error<double>(DimensionsDontMatch);
    
    auto targetClientPtr = target.get().lock();
    if (!targetClientPtr)
      return Error<double>(NoLabelSet);
    auto targetDataSet = targetClientPtr->getLabelSet();
    if (targetDataSet.size() == 0)
      return Error<double>(EmptyLabelSet);
    if (sourceDataSet.size() != targetDataSet.size())
      return Error<double>(SizesDontMatch);

    mAlgorithm.encoder.fit(targetDataSet);

    if (mTracker.changed(get<kHidden>(), get<kActivation>())) {
      mAlgorithm.mlp.init(sourceDataSet.pointSize(),
                          mAlgorithm.encoder.numLabels(), get<kHidden>(),
                          get<kActivation>(), 1);//sigmoid output
    }
    DataSet result(1);
    auto data = sourceDataSet.getData();
    auto tgt = targetDataSet.getData();

    RealMatrix oneHot(targetDataSet.size(), mAlgorithm.encoder.numLabels());
    oneHot.fill(0);
    for (index i = 0; i < targetDataSet.size(); i++) {
      mAlgorithm.encoder.encodeOneHot(tgt.row(i)(0), oneHot.row(i));
    }

    algorithm::SGD sgd;
    double error =
        sgd.train(mAlgorithm.mlp, data, oneHot, get<kIter>(), get<kBatchSize>(),
                  get<kRate>(), get<kMomentum>(), get<kVal>());

    return error;
  }

  MessageResult<void> predict(DataSetClientRef srcClient,
                              LabelSetClientRef destClient) {
    auto srcPtr = srcClient.get().lock();
    auto destPtr = destClient.get().lock();
    if(!srcPtr)return Error(NoDataSet);
    if(!destPtr)return Error(NoLabelSet);
    auto srcDataSet = srcPtr->getDataSet();
    if (srcDataSet.size() == 0)
      return Error(EmptyDataSet);
    if (!mAlgorithm.mlp.trained())
      return Error(NoDataFitted);
    if (srcDataSet.dims() != mAlgorithm.dims())
      return Error(WrongPointSize);

    StringVector ids{srcDataSet.getIds()};
    RealMatrix output(srcDataSet.size(), mAlgorithm.encoder.numLabels());
    mAlgorithm.mlp.process(srcDataSet.getData(), output, 0,  mAlgorithm.mlp.size());
    LabelSet result(1);
    for (index i = 0; i < srcDataSet.size(); i++) {
      StringVector label = {mAlgorithm.encoder.decodeOneHot(output.row(i))};
      result.add(ids(i), label);
    }
    destPtr->setLabelSet(result);
    return OK();
  }

  MessageResult<string> predictPoint(BufferPtr in) {
    if (!in)
      return Error<string>(NoBuffer);
    BufferAdaptor::Access inBuf(in.get());
    if (!inBuf.exists())
      return Error<string>(InvalidBuffer);
    if (inBuf.numFrames() != mAlgorithm.mlp.dims())
      return Error<string>(WrongPointSize);
    if (!mAlgorithm.mlp.trained())
      return Error<string>(NoDataFitted);

    index layer = mAlgorithm.mlp.size();
    RealVector src(mAlgorithm.mlp.dims());
    RealVector dest(mAlgorithm.mlp.outputSize(layer));
    src = inBuf.samps(0, mAlgorithm.mlp.dims(), 0);
    mAlgorithm.mlp.processFrame(src, dest, 0, layer);
    auto label = mAlgorithm.encoder.decodeOneHot(dest);
    return label;
  }

  FLUID_DECLARE_MESSAGES(makeMessage("fit", &MLPClassifierClient::fit),
                         makeMessage("predict", &MLPClassifierClient::predict),
                         makeMessage("predictPoint",
                                     &MLPClassifierClient::predictPoint),
                         makeMessage("clear", &MLPClassifierClient::clear),
                         makeMessage("cols", &MLPClassifierClient::dims),
                         makeMessage("size", &MLPClassifierClient::size),
                         makeMessage("load", &MLPClassifierClient::load),
                         makeMessage("dump", &MLPClassifierClient::dump),
                         makeMessage("write", &MLPClassifierClient::write),
                         makeMessage("read", &MLPClassifierClient::read));

private:
  FluidInputTrigger mTrigger;
  ParameterTrackChanges<IndexVector, index> mTracker;
};

constexpr std::initializer_list<index> MLPClassifierClient::HiddenLayerDefaults;
using RTMLPClassifierClient = ClientWrapper<MLPClassifierClient>;

} // namespace client
} // namespace fluid
