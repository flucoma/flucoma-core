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
namespace lstmforecast {

constexpr auto LSTMForecastParams = defineParameters(
    StringParam<Fixed<true>>("name", "Name"),
    LongParam("maxIter", "Maximum Number of Iterations", 5, Min(1)),
    LongParam("hiddenSize", "Size of Intermediate LSTM layer", 10, Min(1)),
    FloatParam("learnRate", "Learning Rate", 0.01, Min(0.0), Max(1.0)),
    LongParam("batchSize", "Batch Size", 50, Min(1)),
    LongParam("forecastLength", "Length of forecasted data", 0, Min(0)));

class LSTMForecastClient
    : public FluidBaseClient,
      OfflineIn,
      OfflineOut,
      ModelObject,
      public DataClient<algorithm::Recur<algorithm::LSTMCell>>
{
  enum { kName, kIter, kHidden, kRate, kBatch, kForecastLength };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using InputBufferPtr = std::shared_ptr<const BufferAdaptor>;
  using IndexVector = FluidTensor<index, 1>;
  using StringVector = FluidTensor<string, 1>;
  using DataSet = FluidDataSet<string, double, 1>;
  using DataSeries = FluidDataSeries<string, double, 1>;
  using LabelSet = FluidDataSet<string, string, 1>;

  using ParamDescType = decltype(LSTMForecastParams);
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
    return LSTMForecastParams;
  }

  LSTMForecastClient(ParamSetViewType& p, FluidContext&) : mParams(p)
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

  MessageResult<double> fit(InputDataSeriesClientRef dataSeriesClient)
  {
    using namespace algorithm;

    const auto sourceClientPtr = dataSeriesClient.get().lock();
    if (!sourceClientPtr) return Error<double>(NoDataSet);

    const auto sourceDataSeries = sourceClientPtr->getDataSeries();
    if (sourceDataSeries.size() == 0) return Error<double>(EmptyDataSet);

    if (mAlgorithm.initialized())
    {
      if (sourceDataSeries.dims() != mAlgorithm.dims())
        return Error<double>(DimensionsDontMatch);
      if (get<kHidden>() != mAlgorithm.hiddenDims())
        mAlgorithm.init(sourceDataSeries.dims(), get<kHidden>(),
                        sourceDataSeries.dims());
    }
    else
      mAlgorithm.init(sourceDataSeries.dims(), get<kHidden>(),
                      sourceDataSeries.dims());

    auto data = sourceDataSeries.getData();
    return RecurSGD<LSTMCell>{}.trainPredictor(mAlgorithm, data, get<kIter>(),
                                               get<kBatch>(), get<kRate>());
  }

  MessageResult<void> predict(InputDataSeriesClientRef dataSeriesClient,
                              DataSetClientRef         dataSetClient)
  {
    // const auto sourceClientPtr = dataSeriesClient.get().lock();
    // if (!sourceClientPtr) return Error<void>(NoDataSet);

    // const auto sourceDataSeries = sourceClientPtr->getDataSeries();
    // if (sourceDataSeries.size() == 0) return Error<void>(EmptyDataSet);

    // const auto targetClientPtr = dataSetClient.get().lock();
    // if (!targetClientPtr) return Error<void>(NoDataSet);

    // if (!mAlgorithm.trained()) return Error(NoDataFitted);
    // if (sourceDataSeries.dims() != mAlgorithm.dims())
    //   return Error(WrongPointSize);

    // RealMatrix   output(sourceDataSeries.size(), mAlgorithm.size());
    // StringVector ids{sourceDataSeries.getIds()};

    // auto& data = sourceDataSeries.getData();
    // for (index i = 0; i < output.rows(); i++)
    // {
    //   mAlgorithm.reset();
    //   mAlgorithm.process(data[i], output.row(i));
    // }

    // DataSet result(ids, output);
    // targetClientPtr->setDataSet(result);

    return OK();
  }

  MessageResult<void> predictPoint(InputBufferPtr in, BufferPtr out,
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

    RealMatrix dest(forecastLength, mAlgorithm.size());
    RealVector output(mAlgorithm.size()), pred(mAlgorithm.size());

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
        makeMessage("fit", &LSTMForecastClient::fit),
        makeMessage("predict", &LSTMForecastClient::predict),
        makeMessage("predictPoint", &LSTMForecastClient::predictPoint),
        makeMessage("clear", &LSTMForecastClient::clear),
        makeMessage("reset", &LSTMForecastClient::reset),
        makeMessage("print", &LSTMForecastClient::reset),
        makeMessage("load", &LSTMForecastClient::load),
        makeMessage("dump", &LSTMForecastClient::dump),
        makeMessage("write", &LSTMForecastClient::write),
        makeMessage("read", &LSTMForecastClient::read));
  }
};

using LSTMForecastRef = SharedClientRef<const LSTMForecastClient>;

} // namespace lstmforecast

using NRTThreadedLSTMForecastClient =
    NRTThreadingAdaptor<typename lstmforecast::LSTMForecastRef::SharedType>;

} // namespace client
} // namespace fluid