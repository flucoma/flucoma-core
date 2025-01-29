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
#include "../../algorithms/public/DTW.hpp"
#include "../../algorithms/util/FluidEigenMappings.hpp"
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

constexpr auto DTWRegressorParams = defineParameters(
    StringParam<Fixed<true>>("name", "Name"),
    LongParam("numNeighbours", "Number of Nearest Neighbours", 3, Min(1)),
    EnumParam("constraint", "Constraint Type", 0, "Unconstrained", "Ikatura",
              "Sakoe-Chiba"),
    FloatParam("constraintParam", "Sakoe-Chiba radius or Ikatura max gradient",
               3, Min(0)));

class DTWRegressorClient : public FluidBaseClient,
                           OfflineIn,
                           OfflineOut,
                           ModelObject,
                           public DataClient<DTWRegressorData>
{
  enum { kName, kNumNeighbors, kConstraint, kParam };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using InputBufferPtr = std::shared_ptr<const BufferAdaptor>;
  using LabelSet = FluidDataSet<string, string, 1>;
  using DataSet = FluidDataSet<string, double, 1>;
  using StringVector = FluidTensor<string, 1>;

  using ParamDescType = decltype(DTWRegressorParams);

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
    return DTWRegressorParams;
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
    auto dataSeriesClientPtr = dataSeriesClient.get().lock();
    if (!dataSeriesClientPtr) return Error(NoDataSeries);

    auto dataSetPtr = dataSetClient.get().lock();
    if (!dataSetPtr) return Error(NoDataSet);

    auto dataSeries = dataSeriesClientPtr->getDataSeries();
    if (dataSeries.size() == 0) return Error(EmptyDataSeries);

    auto dataSet = dataSetPtr->getDataSet();
    if (dataSet.size() == 0) return Error(EmptyDataSet);

    if (dataSeries.size() != dataSet.size()) return Error(SizesDontMatch);

    auto seriesIds = dataSeries.getIds(), mappingIds = dataSet.getIds();

    bool everySeriesHasALabel = std::is_permutation(
        seriesIds.begin(), seriesIds.end(), mappingIds.begin());

    if (everySeriesHasALabel)
    {
      mAlgorithm.series = dataSeries;
      mAlgorithm.mappings = dataSet;

      return OK();
    }
    else
      return Error(PointNotFound);
  }

  MessageResult<void> predictSeries(InputBufferPtr in, BufferPtr out) const
  {
    if (!in || !out) return Error(NoBuffer);

    BufferAdaptor::ReadAccess inBuf = in.get();
    BufferAdaptor::Access     outBuf = out.get();

    if (!inBuf.exists()) return Error(InvalidBuffer);
    if (!outBuf.exists()) return Error(InvalidBuffer);

    if (inBuf.numChans() < mAlgorithm.series.dims())
      return Error(WrongPointSize);

    Result resizeResult =
        outBuf.resize(mAlgorithm.mappings.dims(), 1, inBuf.sampleRate());
    if (!resizeResult.ok()) return Error(BufferAlloc);

    RealMatrix series(inBuf.numFrames(), inBuf.numChans());
    series <<= inBuf.allFrames().transpose();

    MessageResult<RealVector> result = kNearestWeightedSum(series);
    if (result.ok())
      outBuf.samps(0, result.value().size(), 0) <<= result.value();
    else
      return Error(result.message());

    return OK();
  }

  MessageResult<void> predict(InputDataSeriesClientRef source,
                              DataSetClientRef         dest) const
  {

    auto sourcePtr = source.get().lock();
    if (!sourcePtr) return Error(NoDataSeries);

    auto destPtr = dest.get().lock();
    if (!destPtr) return Error(NoLabelSet);

    auto dataSeries = sourcePtr->getDataSeries();
    if (dataSeries.size() == 0) return Error(EmptyDataSeries);

    if (dataSeries.pointSize() != mAlgorithm.series.dims())
      return Error(WrongPointSize);

    if (mAlgorithm.size() == 0) return Error(NoDataFitted);

    FluidTensorView<string, 1> ids = dataSeries.getIds();
    DataSet                    result(mAlgorithm.mappings.dims());

    for (index i = 0; i < dataSeries.size(); i++)
    {
      MessageResult<RealVector> point =
          kNearestWeightedSum(dataSeries.getSeries(ids[i]));

      if (point.ok())
      {
        RealVector pred = point;
        result.add(ids(i), pred);
      }
      else
        return MessageResult<void>{Result::Status::kError, point.message()};
    }

    destPtr->setDataSet(result);
    return OK();
  }


  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("fit", &DTWRegressorClient::fit),
        makeMessage("predict", &DTWRegressorClient::predict),
        makeMessage("predictSeries", &DTWRegressorClient::predictSeries),
        makeMessage("clear", &DTWRegressorClient::clear),
        makeMessage("size", &DTWRegressorClient::size),
        makeMessage("load", &DTWRegressorClient::load),
        makeMessage("dump", &DTWRegressorClient::dump),
        makeMessage("write", &DTWRegressorClient::write),
        makeMessage("read", &DTWRegressorClient::read));
  }

private:
  MessageResult<RealVector>
  kNearestWeightedSum(InputRealMatrixView series,
                      Allocator&          alloc = FluidDefaultAllocator()) const
  {
    using namespace algorithm::_impl;

    index k = get<kNumNeighbors>();
    if (k < 1) return Error<RealVector>(SmallK);
    if (k > mAlgorithm.size()) return Error<RealVector>(LargeK);

    rt::vector<InputRealMatrixView> ds = mAlgorithm.series.getData();

    if (series.cols() < mAlgorithm.series.dims())
      return Error<RealVector>(WrongPointSize);

    rt::vector<index>  indices(asUnsigned(mAlgorithm.size()));
    rt::vector<double> distances(asUnsigned(mAlgorithm.size()));

    std::iota(indices.begin(), indices.end(), 0);

    algorithm::DTWConstraint constraint =
        (algorithm::DTWConstraint) get<kConstraint>();

    std::transform(
        indices.begin(), indices.end(), distances.begin(),
        [&series, &ds, &constraint, this](index i) {
          double dist =
              mAlgorithm.dtw.process(series, ds[i], constraint, get<kParam>());
          return std::max(std::numeric_limits<double>::epsilon(), dist);
        });

    std::sort(indices.begin(), indices.end(), [&distances](index a, index b) {
      return distances[asUnsigned(a)] < distances[asUnsigned(b)];
    });

    ScopedEigenMap<Eigen::VectorXd> result(mAlgorithm.mappings.dims(), alloc);
    InputRealMatrixView             mappings = mAlgorithm.mappings.getData();

    double totalWeight = 0.0;
    std::for_each(indices.begin(), indices.begin() + get<kNumNeighbors>(),
                  [&](index& i) {
                    double weight = 1.0 / distances[i];

                    totalWeight += weight;
                    result += asEigen<Eigen::Matrix>(mappings.row(i)) * weight;
                  });

    result.noalias() = result * (1 / totalWeight);
    return RealVector(asFluid(result));
  }
};

using DTWRegressorRef = SharedClientRef<const DTWRegressorClient>;

} // namespace dtwclassifier

using NRTThreadedDTWRegressorClient =
    NRTThreadingAdaptor<typename dtwclassifier::DTWRegressorRef::SharedType>;

} // namespace client
} // namespace fluid
