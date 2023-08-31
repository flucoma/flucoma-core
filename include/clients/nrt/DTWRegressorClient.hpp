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
#include "NRTClient.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../../algorithms/public/DTW.hpp"
#include "../../data/FluidDataSeries.hpp"
#include "../../data/FluidDataSet.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"

namespace fluid {
namespace client {
namespace dtwclassifier {

struct DTWRegressorData
{
  algorithm::DTW                          dtw;
  FluidDataSeries<std::string, double, 1> series{0};
  FluidDataSet<std::string, double, 1>    mappings{0};

  index size() const { return series.size(); }
  index dims() const { return series.dims(); }

  void clear()
  {
    mappings = FluidDataSet<std::string, double, 1>(0);
    series = FluidDataSeries<std::string, double, 1>(0);

    dtw.clear();
  }
  bool initialized() const { return dtw.initialized(); }
};

void to_json(nlohmann::json& j, const DTWRegressorData& data)
{
  j["mappings"] = data.mappings;
  j["series"] = data.series;
}

bool check_json(const nlohmann::json& j, const DTWRegressorData&)
{
  return fluid::check_json(j, {"mappings", "series"},
                           {JSONTypes::OBJECT, JSONTypes::OBJECT});
}

void from_json(const nlohmann::json& j, DTWRegressorData& data)
{
  data.series = j.at("series").get<FluidDataSeries<std::string, double, 1>>();
  data.mappings = j.at("mappings").get<FluidDataSet<std::string, double, 1>>();
}

constexpr auto DTWClassifierParams = defineParameters(
    StringParam<Fixed<true>>("name", "Name"),
    LongParam("numNeighbours", "Number of Nearest Neighbours", 3, Min(1)),
    EnumParam("constraint", "Constraint Type", 0, "Unconstrained", "Ikatura",
              "Sakoe-Chiba"),
    FloatParam("radius", "Sakoe-Chiba Constraint Radius", 2, Min(0)),
    FloatParam("gradient", "Ikatura Parallelogram max gradient", 1, Min(1)));

class DTWRegressorClient : public FluidBaseClient,
                           OfflineIn,
                           OfflineOut,
                           ModelObject,
                           public DataClient<DTWRegressorData>
{
  enum { kName, kNumNeighbors, kConstraint, kRadius, kGradient };

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

  DTWRegressorClient(ParamSetViewType& p, FluidContext&) : mParams(p) {}

  template <typename T>
  Result process(FluidContext&)
  {
    return {};
  }

  // not fitting anything, you just set the input series and output labels
  MessageResult<void> fit(InputDataSeriesClientRef dataSeriesClient,
                          InputDataSetClientRef    dataSetClient)
  {
  }

  MessageResult<RealVector> predictPoint(InputBufferPtr data) const
  {
  }

  MessageResult<void> predict(InputDataSeriesClientRef source,
                              DataSetClientRef         dest) const
  {
    return OK();
  }


  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("fit", &DTWRegressorClient::fit),
        makeMessage("predict", &DTWRegressorClient::predict),
        makeMessage("predictPoint", &DTWRegressorClient::predictPoint),
        makeMessage("clear", &DTWRegressorClient::clear),
        makeMessage("size", &DTWRegressorClient::size),
        makeMessage("load", &DTWRegressorClient::load),
        makeMessage("dump", &DTWRegressorClient::dump),
        makeMessage("write", &DTWRegressorClient::write),
        makeMessage("read", &DTWRegressorClient::read));
  }

private:
  float constraintParam(algorithm::DTWConstraint constraint) const
  {
    using namespace algorithm;

    switch (constraint)
    {
    case DTWConstraint::kIkatura: return get<kGradient>();
    case DTWConstraint::kSakoeChiba: return get<kRadius>();
    }

    return 0.0;
  }

  MessageResult<RealVector>
  kNearestWeightedSum(InputRealMatrixView series,
                      Allocator&          alloc = FluidDefaultAllocator()) const
  {
  }
};

using DTWRegressorRef = SharedClientRef<const DTWRegressorClient>;

} // namespace dtwclassifier

using NRTThreadedDTWRegressorClient =
    NRTThreadingAdaptor<typename dtwclassifier::DTWRegressorRef::SharedType>;

} // namespace client
} // namespace fluid
