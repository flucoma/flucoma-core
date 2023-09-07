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

constexpr auto LSTMClassifierParams = defineParameters(
    StringParam<Fixed<true>>("name", "Name"),
    LongParam("maxIter", "Maximum Number of Iterations", 50, Min(1)),
    LongParam("hiddenSize", "Size of Intermediate LSTM layer", 50, Min(1)),
    FloatParam("learnRate", "Learning Rate", 0.01, Min(0.0), Max(1.0)),
    LongParam("batchSize", "Batch Size", 50, Min(1)));

class LSTMClassifierClient : public FluidBaseClient,
                             OfflineIn,
                             OfflineOut,
                             ModelObject,
                             public DataClient<LSTMClassifierData>
{
  enum { kName, kIter, kHidden, kRate, kBatch };

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

  MessageResult<double> fit(InputDataSeriesClientRef dataseriesClient) {}
  MessageResult<double> predict(InputDataSeriesClientRef dataseriesClient) {}
  MessageResult<double> predictPoint(InputDataSeriesClientRef dataseriesClient)
  {}
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

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("fit", &LSTMClassifierClient::fit),
        makeMessage("predict", &LSTMClassifierClient::predict),
        makeMessage("predictPoint", &LSTMClassifierClient::predictPoint));
  }

private:
};

using LSTMClassifierRef = SharedClientRef<const LSTMClassifierClient>;

} // namespace lstmclassifier

using NRTThreadedDTWClient =
    NRTThreadingAdaptor<typename lstmclassifier::LSTMClassifierRef::SharedType>;

} // namespace client
} // namespace fluid