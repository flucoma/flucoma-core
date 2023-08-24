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
#include "LabelSetClient.hpp"
#include "NRTClient.hpp"
#include "../common/SharedClientUtils.hpp"
#include "../../algorithms/public/DataSetIdSequence.hpp"
#include "../../data/FluidDataSeries.hpp"
#include <sstream>
#include <string>

namespace fluid {
namespace client {
namespace dataseries {

enum { kName };

constexpr auto DataSeriesParams = defineParameters(
    StringParam<Fixed<true>>("name", "Name of the DataSeries")
);

class DataSeriesClient : public FluidBaseClient,
                      OfflineIn,
                      OfflineOut,
                      public DataClient<FluidDataSeries<std::string, double, 1>>
{
public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using InputBufferPtr = std::shared_ptr<const BufferAdaptor>;
  using DataSeries = FluidDataSeries<string, double, 1>;
  using LabelSet = FluidDataSet<string, string, 1>;

  template <typename T>
  Result process(FluidContext&)
  {
    return {};
  }

  using ParamDescType = decltype(DataSeriesParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return DataSeriesParams; }

  DataSeriesClient(ParamSetViewType& p, FluidContext&) : mParams(p) {}

  MessageResult<void> addFrame(string id, InputBufferPtr data)
  {
    if (!data) return Error(NoBuffer);

    BufferAdaptor::ReadAccess buf(data.get());
    if (!buf.exists()) return Error(InvalidBuffer);
    if (buf.numFrames() == 0) return Error(EmptyBuffer);

    if (mAlgorithm.size() == 0)
    {
      if (mAlgorithm.dims() != buf.numFrames()) mAlgorithm = DataSeries(buf.numFrames());
    }
    else if (buf.numFrames() != mAlgorithm.dims()) { return Error(WrongPointSize); }

    mAlgorithm.addFrame(id, buf.samps(0, mAlgorithm.dims(), 0));

    return OK();
  }

  MessageResult<void> addSeries(string id, InputBufferPtr data)
  {
    if (!data) return Error(NoBuffer);

    BufferAdaptor::ReadAccess buf(data.get());
    if (!buf.exists()) return Error(InvalidBuffer);
    if (buf.numFrames() == 0) return Error(EmptyBuffer);

    if (mAlgorithm.size() == 0)
    {
      if (mAlgorithm.dims() != buf.numChans()) mAlgorithm = DataSeries(buf.numChans());
    }
    else if (buf.numChans() != mAlgorithm.dims()) { return Error(WrongPointSize); }

    return mAlgorithm.addSeries(id, buf.allFrames().transpose()) 
           ? OK() : Error(DuplicateIdentifier);
  }

  MessageResult<void> getFrame(string id, index time, BufferPtr data) const
  {
    if (!data) return Error(NoBuffer);

    BufferAdaptor::Access buf(data.get());
    if (!buf.exists()) return Error(InvalidBuffer);

    Result resizeResult = buf.resize(mAlgorithm.dims(), 1, buf.sampleRate());
    if (!resizeResult.ok())
      return {resizeResult.status(), resizeResult.message()};

    RealVector point(mAlgorithm.dims());
    point <<= buf.samps(0, mAlgorithm.dims(), 0);

    bool result = mAlgorithm.getFrame(id, time, point);
    if (result)
    {
      buf.samps(0, mAlgorithm.dims(), 0) <<= point;
      return OK();
    }
    else { return Error(PointNotFound); }
  }

  MessageResult<void> getSeries(string id, BufferPtr data) const
  {
    return OK();
  }

  MessageResult<void> updateFrame(string id, index time, InputBufferPtr data)
  {
    if (!data) return Error(NoBuffer);

    BufferAdaptor::ReadAccess buf(data.get());
    if (!buf.exists()) return Error(InvalidBuffer);
    if (buf.numFrames() < mAlgorithm.dims()) return Error(WrongPointSize);

    return mAlgorithm.updateFrame(id, time, buf.samps(0, mAlgorithm.dims(), 0)) ? OK() : Error(PointNotFound);
  }

  MessageResult<void> updateSeries(string id, InputBufferPtr data)
  {
    return OK();
  }

  MessageResult<void> setFrame(string id, index time, InputBufferPtr data)
  {
    if (!data) return Error(NoBuffer);

    { // restrict buffer lock to this scope in case addPoint is called
      BufferAdaptor::ReadAccess buf(data.get());
      if (!buf.exists()) return Error(InvalidBuffer);
      if (buf.numFrames() < mAlgorithm.dims()) return Error(WrongPointSize);

      bool result = mAlgorithm.updateFrame(id, time, buf.samps(0, mAlgorithm.dims(), 0));
      if (result) return OK();
    }

    return addFrame(id, data);
  }

  MessageResult<void> setSeries(string id, InputBufferPtr data)
  {
    return OK();
  }

  MessageResult<void> deleteFrame(string id, index time)
  {
    return mAlgorithm.removeFrame(id, time) ? OK() : Error(PointNotFound);
  }

  MessageResult<void> deleteSeries(string id)
  {
    return mAlgorithm.removeSeries(id) ? OK() : Error(PointNotFound);
  }

  MessageResult<void> merge(SharedClientRef<const DataSeriesClient> dataseriesClient,
                            bool                           overwrite)
  {
    auto dataseriesClientPtr = dataseriesClient.get().lock();
    if (!dataseriesClientPtr) return Error(NoDataSet);

    auto srcDataSeries = dataseriesClientPtr->getDataSeries();
    if (srcDataSeries.size() == 0) return Error(EmptyDataSet);
    if (srcDataSeries.pointSize() != mAlgorithm.pointSize())
      return Error(WrongPointSize);

    auto       ids = srcDataSeries.getIds();

    for (index i = 0; i < srcDataSeries.size(); i++)
    {
      InputRealMatrixView series = srcDataSeries.getSeries(ids(i));
      bool added = mAlgorithm.addSeries(ids(i), series);
      if (!added && overwrite) mAlgorithm.updateSeries(ids(i), series);
    }

    return OK();
  }

  MessageResult<void> getIds(LabelSetClientRef dest)
  {
    auto destPtr = dest.get().lock();
    if (!destPtr) return Error(NoDataSet);
    destPtr->setLabelSet(getIdsLabelSet());

    return OK();
  }

  MessageResult<void> clear()
  {
    mAlgorithm = DataSeries(0);
    return OK();
  }
  
  MessageResult<string> print()
  {
    return "DataSeries " + std::string(get<kName>()) + ": " + mAlgorithm.print();
  }

  const DataSeries getDataSeries() const { return mAlgorithm; }
  void             setDataSeries(DataSeries ds) { mAlgorithm = ds; }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("addFrame",     &DataSeriesClient::addFrame),
        makeMessage("addSeries",    &DataSeriesClient::addSeries),
        makeMessage("getFrame",     &DataSeriesClient::getFrame),
        makeMessage("getSeries",     &DataSeriesClient::getSeries),
        makeMessage("setFrame",     &DataSeriesClient::setFrame),
        makeMessage("setSeries",     &DataSeriesClient::setSeries),
        makeMessage("updateFrame",  &DataSeriesClient::updateFrame),
        makeMessage("updateSeries",  &DataSeriesClient::updateSeries),
        makeMessage("deleteFrame",  &DataSeriesClient::deleteFrame),
        makeMessage("deleteSeries", &DataSeriesClient::deleteFrame),
        makeMessage("merge",        &DataSeriesClient::merge),
        makeMessage("dump",         &DataSeriesClient::dump),
        makeMessage("load",         &DataSeriesClient::load),
        makeMessage("print",        &DataSeriesClient::print),
        makeMessage("size",         &DataSeriesClient::size),
        makeMessage("cols",         &DataSeriesClient::dims),
        makeMessage("clear",        &DataSeriesClient::clear),
        makeMessage("write",        &DataSeriesClient::write),
        makeMessage("read",         &DataSeriesClient::read),
        makeMessage("getIds",       &DataSeriesClient::getIds)
    );
  }

private:
  LabelSet getIdsLabelSet()
  {
    algorithm::DataSetIdSequence seq("", 0, 0);
    FluidTensor<string, 1>       newIds(mAlgorithm.size());
    FluidTensor<string, 2>       labels(mAlgorithm.size(), 1);
    labels.col(0) <<= mAlgorithm.getIds();
    seq.generate(newIds);
    return LabelSet(newIds, labels);
  };
  
  // double distance(FluidTensorView<const double, 1> point1, FluidTensorView<const double, 1> point2) const
  // {    
  //   return std::transform_reduce(point1.begin(), point1.end(), point2.begin(), 0.0, std::plus{}, [](double v1, double v2){
  //     return (v1-v2) * (v1-v2);
  //   });
  // };
};

} // namespace dataset

using DataSeriesClientRef = SharedClientRef<dataseries::DataSeriesClient>;
using InputDataSeriesClientRef = SharedClientRef<const dataseries::DataSeriesClient>;

using NRTThreadedDataSeriesClient =
    NRTThreadingAdaptor<typename DataSeriesClientRef::SharedType>;

} // namespace client
} // namespace fluid
