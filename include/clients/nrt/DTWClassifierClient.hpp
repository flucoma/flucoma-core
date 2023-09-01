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
  FluidDataSeries<std::string, double, 1>   series{0};
  FluidDataSet<std::string, std::string, 1> labels{1};

  index size() const { return series.size(); }
  index dims() const { return series.dims(); }
  void  clear()
  {
    labels = FluidDataSet<std::string, std::string, 1>(1);
    series = FluidDataSeries<std::string, double, 1>(0);

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
    EnumParam("constraint", "Constraint Type", 0, "Unconstrained", "Ikatura",
              "Sakoe-Chiba"),
    FloatParam("radius", "Sakoe-Chiba Constraint Radius", 2, Min(0)),
    FloatParam("gradient", "Ikatura Parallelogram max gradient", 1, Min(1)));

class DTWClassifierClient : public FluidBaseClient,
                            OfflineIn,
                            OfflineOut,
                            ModelObject,
                            public DataClient<DTWClassifierData>
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

  DTWClassifierClient(ParamSetViewType& p, FluidContext&) : mParams(p) {}

  template <typename T>
  Result process(FluidContext&)
  {
    return {};
  }

  // not fitting anything, you just set the input series and output labels
  MessageResult<void> fit(InputDataSeriesClientRef dataSeriesClient,
                          InputLabelSetClientRef   labelSetClient)
  {
    auto dataSeriesClientPtr = dataSeriesClient.get().lock();
    if (!dataSeriesClientPtr) return Error(NoDataSet);

    auto labelSetPtr = labelSetClient.get().lock();
    if (!labelSetPtr) return Error(NoLabelSet);

    auto dataSeries = dataSeriesClientPtr->getDataSeries();
    if (dataSeries.size() == 0) return Error(EmptyDataSet);

    auto labelSet = labelSetPtr->getLabelSet();
    if (labelSet.size() == 0) return Error(EmptyLabelSet);

    if (dataSeries.size() != labelSet.size()) return Error(SizesDontMatch);

    auto seriesIds = dataSeries.getIds(), labelIds = labelSet.getIds();

    bool everySeriesHasALabel = std::is_permutation(
        seriesIds.begin(), seriesIds.end(), labelIds.begin());

    if (everySeriesHasALabel)
    {
      mAlgorithm.series = dataSeries;
      mAlgorithm.labels = labelSet;

      return OK();
    }
    else
      return Error(EmptyLabel);
  }

  MessageResult<string> predictPoint(InputBufferPtr data) const
  {
    BufferAdaptor::ReadAccess buf = data.get();
    RealMatrix                series(buf.numFrames(), buf.numChans());

    if (buf.numChans() < mAlgorithm.dims())
      return Error<string>(WrongPointSize);

    series <<= buf.allFrames().transpose();

    return kNearestModeLabel(series);
  }

  MessageResult<void> predict(InputDataSeriesClientRef source,
                              LabelSetClientRef        dest) const
  {

    auto sourcePtr = source.get().lock();
    if (!sourcePtr) return Error(NoDataSet);

    auto destPtr = dest.get().lock();
    if (!destPtr) return Error(NoLabelSet);

    auto dataSeries = sourcePtr->getDataSeries();
    if (dataSeries.size() == 0) return Error(EmptyDataSet);

    if (dataSeries.pointSize() != mAlgorithm.dims())
      return Error(WrongPointSize);

    if (mAlgorithm.size() == 0) return Error(NoDataFitted);

    FluidTensorView<string, 1> ids = dataSeries.getIds();
    LabelSet                   result(1);

    for (index i = 0; i < dataSeries.size(); i++)
    {
      StringVector label = {kNearestModeLabel(dataSeries.getSeries(ids[i]))};
      result.add(ids(i), label);
    }

    destPtr->setLabelSet(result);
    return OK();
  }


  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("fit", &DTWClassifierClient::fit),
        makeMessage("predict", &DTWClassifierClient::predict),
        makeMessage("predictPoint", &DTWClassifierClient::predictPoint),
        makeMessage("clear", &DTWClassifierClient::clear),
        makeMessage("size", &DTWClassifierClient::size),
        makeMessage("load", &DTWClassifierClient::load),
        makeMessage("dump", &DTWClassifierClient::dump),
        makeMessage("write", &DTWClassifierClient::write),
        makeMessage("read", &DTWClassifierClient::read));
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

  MessageResult<string> kNearestModeLabel(InputRealMatrixView series) const
  {
    index k = get<kNumNeighbors>();
    if (k < 1) return Error<string>(SmallK);
    if (k > mAlgorithm.size()) return Error<string>(LargeK);

    rt::vector<InputRealMatrixView> ds = mAlgorithm.series.getData();

    if (series.cols() < mAlgorithm.dims()) return Error<string>(WrongPointSize);

    rt::vector<index>  indices(asUnsigned(mAlgorithm.size()));
    rt::vector<double> distances(asUnsigned(mAlgorithm.size()));

    std::iota(indices.begin(), indices.end(), 0);

    algorithm::DTWConstraint constraint =
        (algorithm::DTWConstraint) get<kConstraint>();

    std::transform(
        indices.begin(), indices.end(), distances.begin(),
        [&series, &ds, &constraint, this](index i) {
          double dist = mAlgorithm.dtw.process(series, ds[i], constraint,
                                               constraintParam(constraint));
          return std::max(std::numeric_limits<double>::epsilon(), dist);
        });

    std::sort(indices.begin(), indices.end(), [&distances](index a, index b) {
      return distances[asUnsigned(a)] < distances[asUnsigned(b)];
    });

    rt::unordered_map<std::string, double> labelCount;
    FluidTensorView<const std::string, 2>  labels = mAlgorithm.labels.getData();

    std::for_each(indices.begin(), indices.begin() + get<kNumNeighbors>(),
                  [&](index& i) {
                    return labelCount[labels(i, 0)] += 1.0 / distances[i];
                  });

    auto result = std::max_element(
        labelCount.begin(), labelCount.end(),
        [](auto& left, auto& right) { return left.second < right.second; });

    return result->first;
  }
};

using DTWClassifierRef = SharedClientRef<const DTWClassifierClient>;

} // namespace dtwclassifier

using NRTThreadedDTWClassifierClient =
    NRTThreadingAdaptor<typename dtwclassifier::DTWClassifierRef::SharedType>;

} // namespace client
} // namespace fluid
