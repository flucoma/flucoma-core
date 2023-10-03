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
namespace lstmregressor {

constexpr std::initializer_list<index> HiddenLayerDefaults = {10};

constexpr auto LSTMRegressorParams = defineParameters(
    StringParam<Fixed<true>>("name", "Name"),
    LongArrayParam("hiddenLayers", "Hidden Layer Sizes", HiddenLayerDefaults),
    LongParam("maxIter", "Maximum Number of Iterations", 5, Min(1)),
    FloatParam("learnRate", "Learning Rate", 0.01, Min(0.0), Max(1.0)),
    FloatParam("momentum", "Momentum", 0.9, Min(0.0), Max(0.99)),
    LongParam("batchSize", "Batch Size", 50, Min(1)),
    FloatParam("validation", "Validation Amount", 0.2, Min(0), Max(0.9)));

class LSTMRegressorClient
    : public FluidBaseClient,
      OfflineIn,
      OfflineOut,
      ModelObject,
      public DataClient<algorithm::Recur<algorithm::LSTMCell>>
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

  using ParamDescType = decltype(LSTMRegressorParams);
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
    return LSTMRegressorParams;
  }

  LSTMRegressorClient(ParamSetViewType& p, FluidContext&) : mParams(p)
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
    mAlgorithm.clear();
    return OK();
  }

  MessageResult<void> reset()
  {
    mAlgorithm.reset();
    return OK();
  }

  MessageResult<void> write(string fileName)
  {
    if (!mAlgorithm.initialized() || !mAlgorithm.initialized())
      return Error(NoDataFitted);

    return DataClient::write(fileName);
  }

  MessageResult<string> dump()
  {
    if (!mAlgorithm.initialized() || !mAlgorithm.initialized())
      return Error<string>(NoDataFitted);

    return DataClient::dump();
  }

  MessageResult<double> fit(InputDataSeriesClientRef dataSeriesClient,
                            InputDataSetClientRef    dataSetClient)
  {
    using namespace algorithm;

    const auto sourceClientPtr = dataSeriesClient.get().lock();
    if (!sourceClientPtr) return Error<double>(NoDataSet);

    const auto sourceDataSeries = sourceClientPtr->getDataSeries();
    if (sourceDataSeries.size() == 0) return Error<double>(EmptyDataSet);

    const auto targetClientPtr = dataSetClient.get().lock();
    if (!targetClientPtr) return Error<double>(NoLabelSet);

    const auto targetDataSet = targetClientPtr->getDataSet();
    if (targetDataSet.size() == 0) return Error<double>(EmptyDataSet);

    if (sourceDataSeries.size() != targetDataSet.size())
      return Error<double>(SizesDontMatch);

    if (mAlgorithm.initialized() &&
        sourceDataSeries.dims() != mAlgorithm.dims())
      return Error<double>(DimensionsDontMatch);

    if (mTracker.changed(sourceDataSeries.dims(), get<kHidden>(),
                         targetDataSet.pointSize()))
    {
      mAlgorithm.init(sourceDataSeries.dims(), get<kHidden>(),
                      targetDataSet.pointSize());
    }

    auto data = sourceDataSeries.getData();
    auto tgt = targetDataSet.getData();

    return LSTMTrainer().trainManyToOne(mAlgorithm, data, tgt, get<kIter>(),
                                        get<kBatch>(), get<kRate>(),
                                        get<kMomentum>(), get<kValidation>());
  }

  MessageResult<void> predict(InputDataSeriesClientRef dataSeriesClient,
                              DataSetClientRef         dataSetClient)
  {
    const auto sourceClientPtr = dataSeriesClient.get().lock();
    if (!sourceClientPtr) return Error<void>(NoDataSet);

    const auto sourceDataSeries = sourceClientPtr->getDataSeries();
    if (sourceDataSeries.size() == 0) return Error<void>(EmptyDataSet);

    const auto targetClientPtr = dataSetClient.get().lock();
    if (!targetClientPtr) return Error<void>(NoDataSet);

    if (!mAlgorithm.trained()) return Error(NoDataFitted);
    if (sourceDataSeries.dims() != mAlgorithm.dims())
      return Error(WrongPointSize);

    RealMatrix   output(sourceDataSeries.size(), mAlgorithm.outputDims());
    StringVector ids{sourceDataSeries.getIds()};

    auto& data = sourceDataSeries.getData();
    for (index i = 0; i < output.rows(); i++)
    {
      mAlgorithm.reset();
      mAlgorithm.process(data[i], output.row(i));
    }

    DataSet result(ids, output);
    targetClientPtr->setDataSet(result);

    return OK();
  }

  MessageResult<void> predictSeries(InputBufferPtr in, BufferPtr out)
  {
    if (!in || !out) return Error(NoBuffer);

    BufferAdaptor::ReadAccess inBuf(in.get());
    BufferAdaptor::Access     outBuf(out.get());

    if (!inBuf.exists()) return Error(InvalidBuffer);
    if (!outBuf.exists()) return Error(InvalidBuffer);
    if (inBuf.numFrames() == 0) return Error(EmptyBuffer);

    if (!mAlgorithm.trained()) return Error(NoDataFitted);
    if (inBuf.numChans() != mAlgorithm.dims()) return Error(WrongPointSize);

    Result resizeResult =
        outBuf.resize(mAlgorithm.outputDims(), 1, inBuf.sampleRate());

    RealMatrix src(inBuf.numFrames(), inBuf.numChans());
    RealVector dest(mAlgorithm.outputDims());
    src <<= inBuf.allFrames().transpose();

    mAlgorithm.reset();
    mAlgorithm.process(src, dest);

    outBuf.samps(0, dest.size(), 0) <<= dest;

    return OK();
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("fit", &LSTMRegressorClient::fit),
        makeMessage("predict", &LSTMRegressorClient::predict),
        makeMessage("predictSeries", &LSTMRegressorClient::predictSeries),
        makeMessage("clear", &LSTMRegressorClient::clear),
        makeMessage("reset", &LSTMRegressorClient::reset),
        makeMessage("print", &LSTMRegressorClient::reset),
        makeMessage("load", &LSTMRegressorClient::load),
        makeMessage("dump", &LSTMRegressorClient::dump),
        makeMessage("write", &LSTMRegressorClient::write),
        makeMessage("read", &LSTMRegressorClient::read));
  }

private:
  ParameterTrackChanges<index, IndexVector, index> mTracker;
};

using LSTMRegressorRef = SharedClientRef<const LSTMRegressorClient>;

} // namespace lstmregressor

using NRTThreadedLSTMRegressorClient =
    NRTThreadingAdaptor<typename lstmregressor::LSTMRegressorRef::SharedType>;

} // namespace client
} // namespace fluid