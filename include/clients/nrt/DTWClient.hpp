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
#include "NRTClient.hpp"
#include "../../algorithms/public/DTW.hpp"
#include <string>

namespace fluid {
namespace client {
namespace dtw {

constexpr auto DTWParams = defineParameters(
    StringParam<Fixed<true>>("name", "Name"),
    EnumParam("constraint", "Constraint Type", 0, "Unconstrained", "Ikatura",
              "Sakoe-Chiba"),
    LongParam("radius", "Sakoe-Chiba Constraint Radius", 2, Min(0)),
    FloatParam("gradient", "Ikatura Parallelogram max gradient", 1.0,
               Min(1.0)));

class DTWClient : public FluidBaseClient,
                  OfflineIn,
                  OfflineOut,
                  ModelObject,
                  public DataClient<algorithm::DTW>
{
  enum { kName, kConstraint, kRadius, kGradient };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using InputBufferPtr = std::shared_ptr<const BufferAdaptor>;
  using StringVector = FluidTensor<rt::string, 1>;

  using ParamDescType = decltype(DTWParams);
  using ParamSetViewType = ParameterSetView<ParamDescType>;
  using ParamValues = typename ParamSetViewType::ValueTuple;

  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return DTWParams; }

  DTWClient(ParamSetViewType& p, FluidContext&) : mParams(p)
  {
    controlChannelsIn(1);
    controlChannelsOut({1, 1});
  }

  template <typename T>
  Result process(FluidContext&)
  {
    return {};
  }

  MessageResult<double> cost(InputDataSeriesClientRef dataseriesClient,
                             string id1, string id2)
  {
    auto dataseriesClientPtr = dataseriesClient.get().lock();
    if (!dataseriesClientPtr) return Error<double>(NoDataSet);

    auto srcDataSeries = dataseriesClientPtr->getDataSeries();
    if (srcDataSeries.size() == 0) return Error<double>(EmptyDataSet);

    index i1 = srcDataSeries.getIndex(id1), i2 = srcDataSeries.getIndex(id2);

    if (i1 < 0 || i2 < 0) return Error<double>(PointNotFound);

    InputRealMatrixView series1 = srcDataSeries.getSeries(id1),
                        series2 = srcDataSeries.getSeries(id2);

    algorithm::DTWConstraint constraint =
        (algorithm::DTWConstraint) get<kConstraint>();

    return mAlgorithm.process(series1, series2, constraint,
                              constraintParam(constraint));
  }

  MessageResult<double> bufCost(InputBufferPtr data1, InputBufferPtr data2)
  {
    if (!data1 || !data2) return Error<double>(NoBuffer);

    BufferAdaptor::ReadAccess buf1(data1.get()), buf2(data2.get());

    if (!buf1.exists() || !buf2.exists()) return Error<double>(InvalidBuffer);
    if (buf1.numChans() != buf2.numChans())
      return Error<double>(WrongPointSize);
    if (buf1.numFrames() == 0 || buf2.numFrames() == 0)
      return Error<double>(EmptyBuffer);

    RealMatrix buf1frames(buf1.numFrames(), buf1.numChans()),
        buf2frames(buf2.numFrames(), buf2.numChans());

    buf1frames <<= buf1.allFrames().transpose();
    buf2frames <<= buf2.allFrames().transpose();

    algorithm::DTWConstraint constraint =
        (algorithm::DTWConstraint) get<kConstraint>();

    return mAlgorithm.process(buf1frames, buf2frames, constraint,
                              constraintParam(constraint));
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(makeMessage("cost", &DTWClient::cost),
                          makeMessage("bufCost", &DTWClient::bufCost));
  }

private:
  float constraintParam(algorithm::DTWConstraint constraint)
  {
    using namespace algorithm;

    switch (constraint)
    {
    case DTWConstraint::kIkatura: return get<kGradient>();
    case DTWConstraint::kSakoeChiba: return get<kRadius>();
    }

    return 0.0;
  }
};

using DTWRef = SharedClientRef<const DTWClient>;

} // namespace dtw

using NRTThreadedDTWClient =
    NRTThreadingAdaptor<typename dtw::DTWRef::SharedType>;

} // namespace client
} // namespace fluid
