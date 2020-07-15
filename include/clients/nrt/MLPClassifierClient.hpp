#pragma once

#include "DataSetClient.hpp"
#include "LabelSetClient.hpp"
#include "NRTClient.hpp"
#include "algorithms/MLP.hpp"
#include "algorithms/SGD.hpp"
#include "algorithms/LabelSetEncoder.hpp"
#include <string>

namespace fluid {
namespace client {

class MLPClassifierClient : public FluidBaseClient,
                            OfflineIn,
                            OfflineOut,
                            ModelObject,
                            public DataClient<algorithm::MLP> {
  enum { kHidden, kActivation, kIter, kRate, kMomentum, kBatchSize, kVal };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using IndexVector = FluidTensor<index, 1>;
  using StringVector = FluidTensor<string, 1>;
  using DataSet = FluidDataSet<string, double, 1>;
  using LabelSet = FluidDataSet<string, string, 1>;

  template <typename T> Result process(FluidContext &) { return {}; }

  static constexpr std::initializer_list<index> HiddenLayerDefaults = {3, 3};

  FLUID_DECLARE_PARAMS(
      LongArrayParam("hidden", "Hidden Layer Sizes", HiddenLayerDefaults),
      EnumParam("activation", "Activation Function", 1, "Identity", "Sigmoid",
                "ReLU", "Tanh"),
      LongParam("maxIter", "Maximum Number of Iterations", 10000),
      FloatParam("learnRate", "Learning Rate", 0.01, Min(0.0), Max(1.0)),
      FloatParam("momentum", "Momentum", 0.5, Min(0.0), Max(0.99)),
      LongParam("batchSize", "Batch Size", 50),
      FloatParam("validation", "Validation Amount", 0.2, Min(0), Max(0.9)));

  MLPClassifierClient(ParamSetViewType &p) : mParams(p) {}


  MessageResult<double> fit(DataSetClientRef source, LabelSetClientRef target) {
    auto sourceClientPtr = source.get().lock();
    if (!sourceClientPtr)
      return Error<double>(NoDataSet);
    auto sourceDataSet = sourceClientPtr->getDataSet();
    if (sourceDataSet.size() == 0)
      return Error<double>(EmptyDataSet);

    auto targetClientPtr = target.get().lock();
    if (!targetClientPtr)
      return Error<double>(NoLabelSet);
    auto targetDataSet = targetClientPtr->getLabelSet();
    if (targetDataSet.size() == 0)
      return Error<double>(EmptyLabelSet);
    if (sourceDataSet.size() != targetDataSet.size())
      return Error<double>(SizesDontMatch);

    mLabelSetEncoder.fit(targetDataSet);

    if (mTracker.changed(get<kHidden>(), get<kActivation>())) {
      mAlgorithm.init(sourceDataSet.pointSize(), mLabelSetEncoder.numLabels(),
                      get<kHidden>(), get<kActivation>());
    }
    DataSet result(1);
    auto ids = sourceDataSet.getIds();
    auto data = sourceDataSet.getData();
    auto tgt = targetDataSet.getData();

    RealMatrix oneHot(targetDataSet.size(), mLabelSetEncoder.numLabels());
    oneHot.fill(0);
    for(index i = 0; i < targetDataSet.size(); i++){
      mLabelSetEncoder.encodeOneHot(tgt.row(i)(0), oneHot.row(i));
    }

    algorithm::SGD sgd;
    double error =
        sgd.train(mAlgorithm, data, oneHot, get<kIter>(), get<kBatchSize>(),
                  get<kRate>(), get<kMomentum>(), get<kVal>());

    return error;
  }

  MessageResult<void> predict(DataSetClientRef srcClient,
                              LabelSetClientRef destClient) {
    auto srcPtr = srcClient.get().lock();
    auto destPtr = destClient.get().lock();
    if (!srcPtr || !destPtr)
      return Error(NoDataSet);
    auto srcDataSet = srcPtr->getDataSet();
    if (srcDataSet.size() == 0)
      return Error(EmptyDataSet);
    if (!mAlgorithm.trained())
      return Error(NoDataFitted);
    if (srcDataSet.dims() != mAlgorithm.dims())
      return Error(WrongPointSize);

    index layer = mAlgorithm.size() - 1;
    StringVector ids{srcDataSet.getIds()};
    RealMatrix output(srcDataSet.size(), mLabelSetEncoder.numLabels());
    mAlgorithm.process(srcDataSet.getData(), output, layer);
    LabelSet result(1);
    for (index i = 0; i < srcDataSet.size(); i++) {
      StringVector label = {
        mLabelSetEncoder.decodeOneHot(output.row(i))
      };
      result.add(ids(i), label);
    }
    destPtr->setLabelSet(result);
    return OK();
  }

  MessageResult<string> predictPoint(BufferPtr in) {
    if (!in) return Error<string>(NoBuffer);
    BufferAdaptor::Access inBuf(in.get());
    if (!inBuf.exists()) return Error<string>(InvalidBuffer);
    if (inBuf.numFrames() != mAlgorithm.dims()) return Error<string>(WrongPointSize);
    if (!mAlgorithm.trained()) return Error<string>(NoDataFitted);

    index layer = mAlgorithm.size() - 1;
    RealVector src(mAlgorithm.dims());
    RealVector dest(mAlgorithm.outputSize(layer));
    src = inBuf.samps(0, mAlgorithm.dims(), 0);
    mAlgorithm.processFrame(src, dest, layer);
    auto label = mLabelSetEncoder.decodeOneHot(dest);
    return label;
  }

  MessageResult<void> reset() {
    mAlgorithm.reset();
    return OK();
  }


  FLUID_DECLARE_MESSAGES(makeMessage("fit", &MLPClassifierClient::fit),
                         makeMessage("predict", &MLPClassifierClient::predict),
                         makeMessage("predictPoint",
                                     &MLPClassifierClient::predictPoint),
                         makeMessage("reset", &MLPClassifierClient::reset),
                         makeMessage("cols", &MLPClassifierClient::dims),
                         makeMessage("size", &MLPClassifierClient::size),
                         makeMessage("load", &MLPClassifierClient::load),
                         makeMessage("dump", &MLPClassifierClient::dump),
                         makeMessage("write", &MLPClassifierClient::write),
                         makeMessage("read", &MLPClassifierClient::read));

private:
  ParameterTrackChanges<IndexVector, index> mTracker;
  algorithm::LabelSetEncoder mLabelSetEncoder;
};

using NRTThreadedMLPClassifierClient =
    NRTThreadingAdaptor<ClientWrapper<MLPClassifierClient>>;

} // namespace client
} // namespace fluid
