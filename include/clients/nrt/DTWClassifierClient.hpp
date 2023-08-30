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

#include "DataSeriesClient.hpp"
#include "DataSetClient.hpp"
#include "LabelSetClient.hpp"
#include "NRTClient.hpp"
#include "../../algorithms/public/DTW.hpp"
#include "../../data/FluidDataSeries.hpp"
#include "../../data/FluidDataSet.hpp"

namespace fluid {
namespace client {
namespace dtwclassifier {

struct DTWClassifierData
{
  algorithm::DTW                            dtw;
  FluidDataSeries<std::string, double, 1>   series{1};
  FluidDataSet<std::string, std::string, 1> labels{1};

  index size() const { return labels.size(); }
  index dims() const { return dtw.dims(); }
  void  clear()
  {
    labels = FluidDataSet<std::string, std::string, 1>(1);
    series = FluidDataSeries<std::string, double, 1>(1);

    dtw.clear();
  }
  bool initialized() const { return dtw.initialized(); }
};

void to_json(nlohmann::json& j, const DTWClassifierData& data)
{
  j["labels"] = data.labels;
  j["series"] = data.series;
}

bool check_json(const nlohmann::json& j, const DTWClassifierData&)
{
  return fluid::check_json(j, {"labels", "series"},
                           {JSONTypes::OBJECT, JSONTypes::OBJECT});
}

void from_json(const nlohmann::json& j, DTWClassifierData& data)
{
  data.series = j.at("series").get<FluidDataSeries<std::string, double, 1>>();
  data.labels = j.at("labels").get<FluidDataSet<std::string, std::string, 1>>();
}

constexpr auto DTWClassifierParams = defineParameters(
    StringParam<Fixed<true>>("name", "Name"),
    LongParam("numNeighbours", "Number of Nearest Neighbours", 3, Min(1)),
    EnumParam("weight", "Weight Neighbours by Distance", 1, "No", "Yes"));

class DTWClassifierClient : public FluidBaseClient,
                            OfflineIn,
                            OfflineOut,
                            ModelObject,
                            public DataClient<DTWClassifierData>
{
  enum { kName, kNumNeighbors, kWeight };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using InputBufferPtr = std::shared_ptr<const BufferAdaptor>;
  using LabelSet = FluidDataSet<string, string, 1>;
  using DataSet = FluidDataSet<string, double, 1>;
  using StringVector = FluidTensor<string, 1>;

  using ParamDescType = decltype(DTWClassifierParams);

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
    return DTWClassifierParams;
  }

  DTWClassifierClient(ParamSetViewType& p, FluidContext&) : mParams(p) {}

  template <typename T>
  Result process(FluidContext&)
  {
    return {};
  }

  MessageResult<void> fit(InputDataSeriesClientRef dataSeriesClient,
                          InputLabelSetClientRef   labelSetClient)
  {
  }

  MessageResult<string> predictPoint(InputBufferPtr data) const
  {
  }

  MessageResult<void> predict(InputDataSetClientRef source,
                              LabelSetClientRef     dest) const
  {
  }


  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("fit", &DTWClassifierClient::fit),
        makeMessage("predict", &DTWClassifierClient::predict),
        makeMessage("predictPoint", &DTWClassifierClient::predictPoint),
        makeMessage("cols", &DTWClassifierClient::dims),
        makeMessage("clear", &DTWClassifierClient::clear),
        makeMessage("size", &DTWClassifierClient::size),
        makeMessage("load", &DTWClassifierClient::load),
        makeMessage("dump", &DTWClassifierClient::dump),
        makeMessage("write", &DTWClassifierClient::write),
        makeMessage("read", &DTWClassifierClient::read));
  }

};

using DTWClassifierRef = SharedClientRef<const DTWClassifierClient>;

constexpr auto DTWClassifierQueryParams = defineParameters(
    DTWClassifierRef::makeParam("model", "Source model"),
    LongParam("numNeighbours", "Number of Nearest Neighbours", 3, Min(1)),
    EnumParam("weight", "Weight Neighbours by Distance", 1, "No", "Yes"),
    InputBufferParam("inputPointBuffer", "Input Point Buffer"),
    BufferParam("predictionBuffer", "Prediction Buffer"));

class DTWClassifierQuery : public FluidBaseClient, ControlIn, ControlOut
{
  enum { kModel, kNumNeighbors, kWeight, kInputBuffer, kOutputBuffer };

public:
  using ParamDescType = decltype(DTWClassifierQueryParams);

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
    return DTWClassifierQueryParams;
  }

  DTWClassifierQuery(ParamSetViewType& p, FluidContext&) : mParams(p)
  {
    controlChannelsIn(1);
    controlChannelsOut({1, 1});
  }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>>& input,
               std::vector<FluidTensorView<T, 1>>& output, FluidContext& c)
  {
  }

  index latency() { return 0; }
};

} // namespace dtwclassifier

using NRTThreadedDTWClassifierClient =
    NRTThreadingAdaptor<typename knnclassifier::DTWClassifierRef::SharedType>;

using RTDTWClassifierQueryClient =
    ClientWrapper<knnclassifier::DTWClassifierQuery>;
} // namespace client
} // namespace fluid
