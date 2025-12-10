/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "DataClient.hpp"
#include "DataSeriesClient.hpp"
#include "DataSetClient.hpp"
#include "LabelSetClient.hpp"
#include "NRTClient.hpp"
#include "../../algorithms/public/LSTM.hpp"
#include "../../algorithms/public/LabelSetEncoder.hpp"
#include "../../algorithms/public/Recur.hpp"
#include "../../algorithms/public/RecurSGD.hpp"
#include "../../data/FluidJSON.hpp"
#include <string>

namespace fluid {
namespace client {
namespace lstmclassifier {

struct LSTMClassifierData
{
  using LSTMType = algorithm::Recur<algorithm::LSTMCell>;
  using LabelEncoder = algorithm::LabelSetEncoder;

  LSTMType     lstm;
  LabelEncoder encoder;
  index        size() const { return lstm.size(); }
  index        dims() const { return lstm.dims(); }
  void         clear()
  {
    lstm.clear();
    lstm.reset();
    encoder.clear();
  }
  bool initialized() const { return lstm.initialized(); }
};


void to_json(nlohmann::json& j, const LSTMClassifierData& data)
{
  j["lstm"] = data.lstm;
  j["labels"] = data.encoder;
}

bool check_json(const nlohmann::json& j, const LSTMClassifierData&)
{
  return fluid::check_json(j, {"lstm", "labels"},
                           {JSONTypes::OBJECT, JSONTypes::OBJECT});
}

void from_json(const nlohmann::json& j, LSTMClassifierData& data)
{
  j.at("lstm").get_to(data.lstm);
  j.at("labels").get_to(data.encoder);
}

constexpr std::initializer_list<index> HiddenLayerDefaults = {10};

constexpr auto LSTMClassifierParams = defineParameters(
    StringParam<Fixed<true>>("name", "Name"),
    LongArrayParam("hiddenLayers", "Hidden Layer Sizes", HiddenLayerDefaults),
    LongParam("maxIter", "Maximum Number of Iterations", 5, Min(1)),
    FloatParam("learnRate", "Learning Rate", 0.01, Min(0.0), Max(1.0)),
    FloatParam("momentum", "Momentum", 0.9, Min(0.0), Max(0.99)),
    LongParam("batchSize", "Batch Size", 50, Min(1)),
    FloatParam("validation", "Validation Amount", 0.2, Min(0), Max(0.9)));

class LSTMClassifierClient : public FluidBaseClient,
                             OfflineIn,
                             OfflineOut,
                             ModelObject,
                             public DataClient<LSTMClassifierData>
{
  enum { kName, kHidden, kIter, kRate, kMomentum, kBatch, kValidation };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using InputBufferPtr = std::shared_ptr<const BufferAdaptor>;
  using IndexVector = FluidTensor<index, 1>;
  using StringVector = FluidTensor<string, 1>;
  using DataSet = FluidDataSet<string, double, 1>;
  using LabelSet = FluidDataSet<string, string, 1>;

  using ParamDescType = decltype(LSTMClassifierParams);
  using ParamSetViewType = ParameterSetView<ParamDescType>;
  using ParamValues = typename ParamSetViewType::ValueTuple;

  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors()
  {
    return LSTMClassifierParams;
  }

  LSTMClassifierClient(ParamSetViewType& p, FluidContext&) : mParams(p)
  {
    controlChannelsIn(1);
    controlChannelsOut({1, 1});
  }

  template <typename T>
  Result process(FluidContext&)
  {
    return {};
  }

  MessageResult<void> clear()
  {
    mAlgorithm.lstm.clear();
    mAlgorithm.encoder.clear();
    return OK();
  }

  MessageResult<void> reset()
  {
    mAlgorithm.lstm.reset();
    return OK();
  }

  MessageResult<void> write(string fileName)
  {
    if (!mAlgorithm.lstm.initialized() || !mAlgorithm.encoder.initialized())
      return Error(NoDataFitted);

    return DataClient::write(fileName);
  }

  MessageResult<string> dump()
  {
    if (!mAlgorithm.lstm.initialized() || !mAlgorithm.encoder.initialized())
      return Error<string>(NoDataFitted);

    return DataClient::dump();
  }

  MessageResult<double> fit(InputDataSeriesClientRef dataSeriesClient,
                            InputLabelSetClientRef   labelSetClient)
  {
    using namespace algorithm;

    const auto sourceClientPtr = dataSeriesClient.get().lock();
    if (!sourceClientPtr) return Error<double>(NoDataSet);

    const auto sourceDataSeries = sourceClientPtr->getDataSeries();
    if (sourceDataSeries.size() == 0) return Error<double>(EmptyDataSet);

    const auto targetClientPtr = labelSetClient.get().lock();
    if (!targetClientPtr) return Error<double>(NoLabelSet);

    const auto targetLabelSet = targetClientPtr->getLabelSet();
    if (targetLabelSet.size() == 0) return Error<double>(EmptyLabelSet);

    if (sourceDataSeries.size() != targetLabelSet.size())
      return Error<double>(SizesDontMatch);

    if (mAlgorithm.initialized() &&
        sourceDataSeries.dims() != mAlgorithm.dims())
      return Error<double>(DimensionsDontMatch);

    mAlgorithm.encoder.fit(targetLabelSet);

    if (mTracker.changed(sourceDataSeries.dims(),
                         mAlgorithm.encoder.numLabels(), get<kHidden>()))
    {
      mAlgorithm.lstm.init(sourceDataSeries.dims(), get<kHidden>(),
                           mAlgorithm.encoder.numLabels());
    }

    mAlgorithm.lstm.setTrained(false);

    auto data = sourceDataSeries.getData();
    auto tgt = targetLabelSet.getData();
    auto ids = sourceDataSeries.getIds();

    RealMatrix oneHot(targetLabelSet.size(), mAlgorithm.encoder.numLabels());
    oneHot.fill(0);

    for (index i = 0; i < targetLabelSet.size(); i++)
    {
      index id = targetLabelSet.getIndex(ids[i]);
      if (id < 0) return Error<double>(PointNotFound);
      mAlgorithm.encoder.encodeOneHot(tgt.row(id)(0), oneHot.row(i));
    }

    return LSTMTrainer().trainManyToOne(
        mAlgorithm.lstm, data, oneHot, get<kIter>(), get<kBatch>(),
        get<kRate>(), get<kMomentum>(), get<kValidation>());
  }

  MessageResult<void> predict(InputDataSeriesClientRef dataSeriesClient,
                              LabelSetClientRef        labelSetClient)
  {
    const auto sourceClientPtr = dataSeriesClient.get().lock();
    if (!sourceClientPtr) return Error<void>(NoDataSet);

    const auto sourceDataSeries = sourceClientPtr->getDataSeries();
    if (sourceDataSeries.size() == 0) return Error<void>(EmptyDataSet);

    const auto targetClientPtr = labelSetClient.get().lock();
    if (!targetClientPtr) return Error<void>(NoLabelSet);

    if (!mAlgorithm.lstm.trained()) return Error(NoDataFitted);
    if (sourceDataSeries.dims() != mAlgorithm.dims())
      return Error(WrongPointSize);

    RealMatrix output(sourceDataSeries.size(), mAlgorithm.encoder.numLabels());

    auto& data = sourceDataSeries.getData();
    for (index i = 0; i < output.rows(); i++)
    {
      mAlgorithm.lstm.reset();
      mAlgorithm.lstm.process(data[i], output.row(i));
    }

    LabelSet     result(1);
    StringVector ids{sourceDataSeries.getIds()};
    for (index i = 0; i < output.size(); i++)
    {
      StringVector label = {mAlgorithm.encoder.decodeOneHot(output.row(i))};
      result.add(ids(i), label);
    }

    targetClientPtr->setLabelSet(result);
    return OK();
  }

  MessageResult<string> predictSeries(InputBufferPtr buffer)
  {
    if (!buffer) return Error<string>(NoBuffer);
    BufferAdaptor::ReadAccess inBuf(buffer.get());
    if (!inBuf.exists()) return Error<string>(InvalidBuffer);
    if (inBuf.numFrames() == 0) return Error<string>(EmptyBuffer);

    if (!mAlgorithm.lstm.trained()) return Error<string>(NoDataFitted);
    if (inBuf.numChans() != mAlgorithm.lstm.dims())
      return Error<string>(WrongPointSize);

    RealMatrix src(inBuf.numFrames(), inBuf.numChans());
    src <<= inBuf.allFrames().transpose();

    RealVector dest(mAlgorithm.lstm.outputDims());
    mAlgorithm.lstm.reset();
    mAlgorithm.lstm.process(src, dest);

    auto& label = mAlgorithm.encoder.decodeOneHot(dest);
    return label;
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("fit", &LSTMClassifierClient::fit),
        makeMessage("predict", &LSTMClassifierClient::predict),
        makeMessage("predictSeries", &LSTMClassifierClient::predictSeries),
        makeMessage("clear", &LSTMClassifierClient::clear),
        makeMessage("reset", &LSTMClassifierClient::reset),
        makeMessage("print", &LSTMClassifierClient::reset),
        makeMessage("load", &LSTMClassifierClient::load),
        makeMessage("dump", &LSTMClassifierClient::dump),
        makeMessage("write", &LSTMClassifierClient::write),
        makeMessage("read", &LSTMClassifierClient::read));
  }

private:
  ParameterTrackChanges<index, index, IndexVector> mTracker;
};

using LSTMClassifierRef = SharedClientRef<const LSTMClassifierClient>;

} // namespace lstmclassifier

using NRTThreadedLSTMClassifierClient =
    NRTThreadingAdaptor<typename lstmclassifier::LSTMClassifierRef::SharedType>;

} // namespace client
} // namespace fluid