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
#include "DataSetClient.hpp"
#include "LabelSetClient.hpp"
#include "NRTClient.hpp"
#include "../common/SharedClientUtils.hpp"
#include "../../algorithms/public/DTW.hpp"
#include "../../algorithms/public/DataSetIdSequence.hpp"
#include "../../data/FluidDataSeries.hpp"
#include <sstream>
#include <string>

namespace fluid {
namespace client {
namespace dataseries {

enum { kName };

constexpr auto DataSeriesParams = defineParameters(
    StringParam<Fixed<true>>("name", "Name of the DataSeries"));

class DataSeriesClient
    : public FluidBaseClient,
      OfflineIn,
      OfflineOut,
      public DataClient<FluidDataSeries<std::string, double, 1>>
{
public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using InputBufferPtr = std::shared_ptr<const BufferAdaptor>;
  using DataSeries = FluidDataSeries<string, double, 1>;
  using DataSet = FluidDataSet<string, double, 1>;
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
      if (mAlgorithm.dims() != buf.numFrames())
        mAlgorithm = DataSeries(buf.numFrames());
    }
    else if (buf.numFrames() != mAlgorithm.dims())
    {
      return Error(WrongPointSize);
    }

    RealVector frame(buf.numFrames());
    frame <<= buf.samps(0, mAlgorithm.dims(), 0);

    mAlgorithm.addFrame(id, frame);

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
      if (mAlgorithm.dims() != buf.numChans())
        mAlgorithm = DataSeries(buf.numChans());
    }
    else if (buf.numChans() != mAlgorithm.dims())
    {
      return Error(WrongPointSize);
    }

    RealMatrix series(buf.numFrames(), buf.numChans());
    series <<= buf.allFrames().transpose();

    return mAlgorithm.addSeries(id, series) ? OK() : Error(DuplicateIdentifier);
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
    if (!data) return Error(NoBuffer);

    BufferAdaptor::Access buf(data.get());
    if (!buf.exists()) return Error(InvalidBuffer);

    index numFrames = mAlgorithm.getNumFrames(id);
    if (numFrames < 0) return Error(PointNotFound);

    Result resizeResult =
        buf.resize(numFrames, mAlgorithm.dims(), buf.sampleRate());
    if (!resizeResult.ok())
      return {resizeResult.status(), resizeResult.message()};

    RealMatrix point(numFrames, mAlgorithm.dims());
    bool       result = mAlgorithm.getSeries(id, point);
    if (result)
    {
      buf.allFrames() <<= point.transpose();
      return OK();
    }
    else { return Error(PointNotFound); }
  }

  MessageResult<void> updateFrame(string id, index time, InputBufferPtr data)
  {
    if (!data) return Error(NoBuffer);

    BufferAdaptor::ReadAccess buf(data.get());
    if (!buf.exists()) return Error(InvalidBuffer);
    if (buf.numFrames() < mAlgorithm.dims()) return Error(WrongPointSize);

    RealVector frame(buf.numFrames());
    frame <<= buf.samps(0, mAlgorithm.dims(), 0);

    return mAlgorithm.updateFrame(id, time, frame) ? OK()
                                                   : Error(PointNotFound);
  }

  MessageResult<void> updateSeries(string id, InputBufferPtr data)
  {
    if (!data) return Error(NoBuffer);

    BufferAdaptor::ReadAccess buf(data.get());
    if (!buf.exists()) return Error(InvalidBuffer);
    if (buf.numChans() < mAlgorithm.dims()) return Error(WrongPointSize);

    RealMatrix series(buf.numFrames(), buf.numChans());
    series <<= buf.allFrames().transpose();

    return mAlgorithm.updateSeries(id, series) ? OK() : Error(PointNotFound);
  }

  MessageResult<void> setFrame(string id, index time, InputBufferPtr data)
  {
    if (!data) return Error(NoBuffer);

    { // restrict buffer lock to this scope in case addPoint is called
      BufferAdaptor::ReadAccess buf(data.get());
      if (!buf.exists()) return Error(InvalidBuffer);
      if (buf.numFrames() < mAlgorithm.dims()) return Error(WrongPointSize);

      RealVector frame(buf.numFrames());
      frame <<= buf.samps(0, mAlgorithm.dims(), 0);

      bool result = mAlgorithm.updateFrame(id, time, frame);
      if (result) return OK();
    }

    return addFrame(id, data);
  }

  MessageResult<void> setSeries(string id, InputBufferPtr data)
  {
    if (!data) return Error(NoBuffer);

    { // restrict buffer lock to this scope in case addPoint is called
      BufferAdaptor::ReadAccess buf(data.get());
      if (!buf.exists()) return Error(InvalidBuffer);
      if (buf.numChans() < mAlgorithm.dims()) return Error(WrongPointSize);

      RealMatrix series(buf.numFrames(), buf.numChans());
      series <<= buf.allFrames().transpose();

      bool result = mAlgorithm.updateSeries(id, series);
      if (result) return OK();
    }

    return addSeries(id, data);
  }

  MessageResult<void> deleteFrame(string id, index time)
  {
    return mAlgorithm.removeFrame(id, time) ? OK() : Error(PointNotFound);
  }

  MessageResult<void> deleteSeries(string id)
  {
    return mAlgorithm.removeSeries(id) ? OK() : Error(PointNotFound);
  }

  MessageResult<void>
  merge(SharedClientRef<const DataSeriesClient> dataseriesClient,
        bool                                    overwrite)
  {
    auto dataseriesClientPtr = dataseriesClient.get().lock();
    if (!dataseriesClientPtr) return Error(NoDataSeries);

    auto srcDataSeries = dataseriesClientPtr->getDataSeries();
    if (srcDataSeries.size() == 0) return Error(EmptyDataSeries);
    if (srcDataSeries.pointSize() != mAlgorithm.pointSize())
      return Error(WrongPointSize);

    auto ids = srcDataSeries.getIds();

    for (index i = 0; i < srcDataSeries.size(); i++)
    {
      InputRealMatrixView series = srcDataSeries.getSeries(ids(i));
      bool                added = mAlgorithm.addSeries(ids(i), series);
      if (!added && overwrite) mAlgorithm.updateSeries(ids(i), series);
    }

    return OK();
  }

  MessageResult<void> getDataSet(DataSetClientRef dest, index time) const
  {
    auto destPtr = dest.get().lock();
    if (!destPtr) return Error(NoDataSet);
    destPtr->setDataSet(getSliceDataSet(time));

    if (destPtr->size() == 0) return Error(EmptyDataSet);

    return OK();
  }

  MessageResult<void> getIds(LabelSetClientRef dest)
  {
    auto destPtr = dest.get().lock();
    if (!destPtr) return Error(NoLabelSet);
    destPtr->setLabelSet(getIdsLabelSet());

    return OK();
  }

  MessageResult<FluidTensor<rt::string, 1>> kNearest(InputBufferPtr data,
                                                     index nNeighbours) const
  {
    // check for nNeighbours > 0 and < size of DS
    if (nNeighbours > mAlgorithm.size())
      return Error<FluidTensor<rt::string, 1>>(SmallDataSet);
    if (nNeighbours <= 0) return Error<FluidTensor<rt::string, 1>>(SmallK);

    BufferAdaptor::ReadAccess buf(data.get());
    if (!buf.exists()) return Error<FluidTensor<rt::string, 1>>(InvalidBuffer);
    if (buf.numChans() < mAlgorithm.dims())
      return Error<FluidTensor<rt::string, 1>>(WrongPointSize);

    FluidTensor<const double, 2> series(buf.allFrames().transpose());

    rt::vector<index>  indices(asUnsigned(mAlgorithm.size()));
    rt::vector<double> distances(asUnsigned(mAlgorithm.size()));

    std::iota(indices.begin(), indices.end(), 0);

    auto ds = mAlgorithm.getData();

    std::transform(
        indices.begin(), indices.end(), distances.begin(),
        [&series, &ds, this](index i) { return distance(series, ds[i], 2); });

    std::sort(indices.begin(), indices.end(), [&distances](index a, index b) {
      return distances[asUnsigned(a)] < distances[asUnsigned(b)];
    });

    FluidTensor<rt::string, 1> labels(nNeighbours);

    std::transform(
        indices.begin(), indices.begin() + nNeighbours, labels.begin(),
        [this](index i) {
          std::string const& id = mAlgorithm.getIds()[i];
          return rt::string{id, 0, id.size(), FluidDefaultAllocator()};
        });

    return labels;
  }

  MessageResult<FluidTensor<double, 1>> kNearestDist(InputBufferPtr data,
                                                     index nNeighbours) const
  {
    // check for nNeighbours > 0 and < size of DS
    if (nNeighbours > mAlgorithm.size())
      return Error<FluidTensor<double, 1>>(SmallDataSet);
    if (nNeighbours <= 0) return Error<FluidTensor<double, 1>>(SmallK);

    BufferAdaptor::ReadAccess buf(data.get());
    if (!buf.exists()) return Error<FluidTensor<double, 1>>(InvalidBuffer);
    if (buf.numChans() < mAlgorithm.dims())
      return Error<FluidTensor<double, 1>>(WrongPointSize);

    FluidTensor<const double, 2> series(buf.allFrames().transpose());

    rt::vector<index>  indices(asUnsigned(mAlgorithm.size()));
    rt::vector<double> distances(asUnsigned(mAlgorithm.size()));

    std::iota(indices.begin(), indices.end(), 0);

    auto ds = mAlgorithm.getData();

    std::transform(
        indices.begin(), indices.end(), distances.begin(),
        [&series, &ds, this](index i) { return distance(series, ds[i], 2); });

    std::sort(indices.begin(), indices.end(), [&distances](index a, index b) {
      return distances[asUnsigned(a)] < distances[asUnsigned(b)];
    });

    FluidTensor<double, 1> labels(nNeighbours);

    std::transform(indices.begin(), indices.begin() + nNeighbours,
                   labels.begin(),
                   [&distances](index i) { return distances[i]; });

    return labels;
  }

  MessageResult<void> clear()
  {
    mAlgorithm = DataSeries(0);
    return OK();
  }

  MessageResult<string> print()
  {
    return "DataSeries " + std::string(get<kName>()) + ": " +
           mAlgorithm.print();
  }

  const DataSeries getDataSeries() const { return mAlgorithm; }
  void             setDataSeries(DataSeries ds) { mAlgorithm = ds; }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("addFrame", &DataSeriesClient::addFrame),
        makeMessage("addSeries", &DataSeriesClient::addSeries),
        makeMessage("getFrame", &DataSeriesClient::getFrame),
        makeMessage("getSeries", &DataSeriesClient::getSeries),
        makeMessage("setFrame", &DataSeriesClient::setFrame),
        makeMessage("setSeries", &DataSeriesClient::setSeries),
        makeMessage("updateFrame", &DataSeriesClient::updateFrame),
        makeMessage("updateSeries", &DataSeriesClient::updateSeries),
        makeMessage("deleteFrame", &DataSeriesClient::deleteFrame),
        makeMessage("deleteSeries", &DataSeriesClient::deleteSeries),
        makeMessage("merge", &DataSeriesClient::merge),
        makeMessage("dump", &DataSeriesClient::dump),
        makeMessage("load", &DataSeriesClient::load),
        makeMessage("print", &DataSeriesClient::print),
        makeMessage("size", &DataSeriesClient::size),
        makeMessage("cols", &DataSeriesClient::dims),
        makeMessage("clear", &DataSeriesClient::clear),
        makeMessage("write", &DataSeriesClient::write),
        makeMessage("read", &DataSeriesClient::read),
        makeMessage("kNearest", &DataSeriesClient::kNearest),
        makeMessage("kNearestDist", &DataSeriesClient::kNearestDist),
        makeMessage("getIds", &DataSeriesClient::getIds),
        makeMessage("getDataSet", &DataSeriesClient::getDataSet));
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

  DataSet getSliceDataSet(index time) const
  {
    DataSet                         ds(mAlgorithm.dims());
    decltype(mAlgorithm)::FrameType frame(mAlgorithm.dims());

    for (auto id : mAlgorithm.getIds())
    {
      bool ret = mAlgorithm.getFrame(id, time, frame);
      if (ret) ds.add(id, frame);
    }

    return ds;
  }

  double distance(InputRealMatrixView x1, InputRealMatrixView x2, index p) const
  {
    algorithm::DTW dtw;
    return dtw.process(x1, x2, algorithm::DTWConstraint::kSakoeChiba,
                       std::min(x1.size(), x2.size()) / 4);
  }
};

} // namespace dataseries

using DataSeriesClientRef = SharedClientRef<dataseries::DataSeriesClient>;
using InputDataSeriesClientRef =
    SharedClientRef<const dataseries::DataSeriesClient>;

using NRTThreadedDataSeriesClient =
    NRTThreadingAdaptor<typename DataSeriesClientRef::SharedType>;

} // namespace client
} // namespace fluid
