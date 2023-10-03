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
namespace lstmforecaster {

constexpr std::initializer_list<index> HiddenLayerDefaults = {10};

constexpr auto LSTMForecasterParams = defineParameters(
    StringParam<Fixed<true>>("name", "Name"),
    LongArrayParam("hiddenLayers", "Hidden Layer Sizes", HiddenLayerDefaults),
    LongParam("maxIter", "Maximum Number of Iterations", 5, Min(1)),
    FloatParam("learnRate", "Learning Rate", 0.01, Min(0.0), Max(1.0)),
    FloatParam("momentum", "Momentum", 0.9, Min(0.0), Max(0.99)),
    LongParam("batchSize", "Batch Size", 50, Min(1)),
    FloatParam("validation", "Validation Amount", 0.2, Min(0), Max(0.9)),
    LongParam("forecastLength", "Length of forecasted data", 0, Min(0)));

class LSTMForecasterClient
    : public FluidBaseClient,
      OfflineIn,
      OfflineOut,
      ModelObject,
      public DataClient<algorithm::Recur<algorithm::LSTMCell>>
{
  enum {
    kName,
    kHidden,
    kIter,
    kRate,
    kMomentum,
    kBatch,
    kValidation,
    kForecastLength
  };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using InputBufferPtr = std::shared_ptr<const BufferAdaptor>;
  using IndexVector = FluidTensor<index, 1>;
  using StringVector = FluidTensor<string, 1>;
  using DataSet = FluidDataSet<string, double, 1>;
  using DataSeries = FluidDataSeries<string, double, 1>;
  using LabelSet = FluidDataSet<string, string, 1>;

  using ParamDescType = decltype(LSTMForecasterParams);
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
    return LSTMForecasterParams;
  }

  LSTMForecasterClient(ParamSetViewType& p, FluidContext&) : mParams(p)
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

  MessageResult<double> fit(InputDataSeriesClientRef dataSeriesClient)
  {
    using namespace algorithm;

    const auto sourceClientPtr = dataSeriesClient.get().lock();
    if (!sourceClientPtr) return Error<double>(NoDataSet);

    const auto sourceDataSeries = sourceClientPtr->getDataSeries();
    if (sourceDataSeries.size() == 0) return Error<double>(EmptyDataSet);

    if (mAlgorithm.initialized() &&
        sourceDataSeries.dims() != mAlgorithm.dims())
      return Error<double>(DimensionsDontMatch);

    if (mTracker.changed(sourceDataSeries.dims(), get<kHidden>(),
                         sourceDataSeries.dims()))
    {
      mAlgorithm.init(sourceDataSeries.dims(), get<kHidden>(),
                      sourceDataSeries.dims());
    }

    auto data = sourceDataSeries.getData();

    return LSTMTrainer().trainPredictor(mAlgorithm, data, get<kIter>(),
                                        get<kBatch>(), get<kRate>(),
                                        get<kMomentum>(), get<kValidation>());
  }

  MessageResult<void> predict(InputDataSeriesClientRef sourceDataSeriesClient,
                              DataSeriesClientRef      targetDataSeriesClient,
                              index forecastLengthOverride = -1)
  {
    assert(mAlgorithm.inputDims() == mAlgorithm.outputDims());

    index forecastLength = forecastLengthOverride < 0 ? get<kForecastLength>()
                                                      : forecastLengthOverride;

    const auto sourceClientPtr = sourceDataSeriesClient.get().lock();
    if (!sourceClientPtr) return Error<void>(NoDataSet);

    const auto sourceDataSeries = sourceClientPtr->getDataSeries();
    if (sourceDataSeries.size() == 0) return Error<void>(EmptyDataSet);

    const auto targetClientPtr = targetDataSeriesClient.get().lock();
    if (!targetClientPtr) return Error<void>(NoDataSet);

    if (!mAlgorithm.trained()) return Error(NoDataFitted);
    if (sourceDataSeries.dims() != mAlgorithm.dims())
      return Error(WrongPointSize);

    StringVector ids{sourceDataSeries.getIds()};
    DataSeries   result(mAlgorithm.outputDims());
    RealVector   output(mAlgorithm.outputDims()), pred(mAlgorithm.outputDims());

    auto& data = sourceDataSeries.getData();
    for (index i = 1; i < sourceDataSeries.size(); i++)
    {
      index thisLength = forecastLength > 0 ? forecastLength : data[i].rows();

      mAlgorithm.reset();
      mAlgorithm.process(data[i], output);

      for (index f = 0; f < thisLength; f++)
      {
        mAlgorithm.processFrame(output, pred);
        result.addFrame(ids[i], pred);
      }
    }

    targetClientPtr->setDataSeries(result);

    return OK();
  }

  MessageResult<void> predictSeries(InputBufferPtr in, BufferPtr out,
                                    index forecastLengthOverride = -1)
  {

    index forecastLength = forecastLengthOverride < 0 ? get<kForecastLength>()
                                                      : forecastLengthOverride;

    if (!in || !out) return Error(NoBuffer);
    BufferAdaptor::ReadAccess inBuf(in.get());
    BufferAdaptor::Access     outBuf(out.get());

    if (!inBuf.exists()) return Error(InvalidBuffer);
    if (!outBuf.exists()) return Error(InvalidBuffer);
    if (inBuf.numFrames() == 0) return Error(EmptyBuffer);

    if (!mAlgorithm.trained()) return Error(NoDataFitted);
    if (inBuf.numChans() != mAlgorithm.dims()) return Error(WrongPointSize);

    forecastLength = forecastLength > 0 ? forecastLength : inBuf.numFrames();
    Result resizeResult =
        outBuf.resize(forecastLength, inBuf.numChans(), inBuf.sampleRate());

    RealMatrix src(inBuf.numFrames(), inBuf.numChans());
    src <<= inBuf.allFrames().transpose();

    RealMatrix dest(forecastLength, mAlgorithm.outputDims());
    RealVector output(mAlgorithm.outputDims()), pred(mAlgorithm.outputDims());

    mAlgorithm.reset();
    mAlgorithm.process(src, output);

    for (index i = 0; i < forecastLength; i++)
    {
      mAlgorithm.processFrame(output, pred);
      dest.row(i) <<= pred;
    }

    outBuf.allFrames().transpose() <<= dest;

    return OK();
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("fit", &LSTMForecasterClient::fit),
        makeMessage("predict", &LSTMForecasterClient::predict),
        makeMessage("predictSeries", &LSTMForecasterClient::predictSeries),
        makeMessage("clear", &LSTMForecasterClient::clear),
        makeMessage("reset", &LSTMForecasterClient::reset),
        makeMessage("print", &LSTMForecasterClient::reset),
        makeMessage("load", &LSTMForecasterClient::load),
        makeMessage("dump", &LSTMForecasterClient::dump),
        makeMessage("write", &LSTMForecasterClient::write),
        makeMessage("read", &LSTMForecasterClient::read));
  }

private:
  ParameterTrackChanges<index, IndexVector, index> mTracker;
};

using LSTMForecasterRef = SharedClientRef<const LSTMForecasterClient>;

} // namespace lstmforecaster

using NRTThreadedLSTMForecasterClient =
    NRTThreadingAdaptor<typename lstmforecaster::LSTMForecasterRef::SharedType>;

} // namespace client
} // namespace fluid