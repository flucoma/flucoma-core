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
#include "../../data/FluidDataSet.hpp"
#include <sstream>
#include <string>

namespace fluid {
namespace client {
namespace dataset {

enum { kName };

constexpr auto DataSetParams =
    defineParameters(StringParam<Fixed<true>>("name", "Name of the DataSet"));

class DataSetClient : public FluidBaseClient,
                      OfflineIn,
                      OfflineOut,
                      public DataClient<FluidDataSet<std::string, double, 1>>
{
public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using InputBufferPtr = std::shared_ptr<const BufferAdaptor>;
  using DataSet = FluidDataSet<string, double, 1>;
  using LabelSet = FluidDataSet<string, string, 1>;

  template <typename T>
  Result process(FluidContext&)
  {
    return {};
  }

  using ParamDescType = decltype(DataSetParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return DataSetParams; }

  DataSetClient(ParamSetViewType& p, FluidContext&) : mParams(p) {}

  MessageResult<void> addPoint(string id, InputBufferPtr data)
  {
    DataSet& dataset = mAlgorithm;
    if (!data) return Error(NoBuffer);
    BufferAdaptor::ReadAccess buf(data.get());
    if (!buf.exists()) return Error(InvalidBuffer);
    if (buf.numFrames() == 0) return Error(EmptyBuffer);
    if (dataset.size() == 0)
    {
      if (dataset.dims() != buf.numFrames()) dataset = DataSet(buf.numFrames());
    }
    else if (buf.numFrames() != dataset.dims())
      return Error(WrongPointSize);
    RealVector point(dataset.dims());
    point <<= buf.samps(0, dataset.dims(), 0);
    return dataset.add(id, point) ? OK() : Error(DuplicateIdentifier);
  }

  MessageResult<void> getPoint(string id, BufferPtr data) const
  {
    if (!data) return Error(NoBuffer);
    BufferAdaptor::Access buf(data.get());
    if (!buf.exists()) return Error(InvalidBuffer);
    Result resizeResult = buf.resize(mAlgorithm.dims(), 1, buf.sampleRate());
    if (!resizeResult.ok())
      return {resizeResult.status(), resizeResult.message()};
    RealVector point(mAlgorithm.dims());
    point <<= buf.samps(0, mAlgorithm.dims(), 0);
    bool result = mAlgorithm.get(id, point);
    if (result)
    {
      buf.samps(0, mAlgorithm.dims(), 0) <<= point;
      return OK();
    }
    else
    {
      return Error(PointNotFound);
    }
  }

  MessageResult<void> updatePoint(string id, InputBufferPtr data)
  {
    if (!data) return Error(NoBuffer);
    BufferAdaptor::ReadAccess buf(data.get());
    if (!buf.exists()) return Error(InvalidBuffer);
    if (buf.numFrames() < mAlgorithm.dims()) return Error(WrongPointSize);
    RealVector point(mAlgorithm.dims());
    point <<= buf.samps(0, mAlgorithm.dims(), 0);
    return mAlgorithm.update(id, point) ? OK() : Error(PointNotFound);
  }

  MessageResult<void> setPoint(string id, InputBufferPtr data)
  {
    if (!data) return Error(NoBuffer);

    { // restrict buffer lock to this scope in case addPoint is called
      BufferAdaptor::ReadAccess buf(data.get());
      if (!buf.exists()) return Error(InvalidBuffer);
      if (buf.numFrames() < mAlgorithm.dims()) return Error(WrongPointSize);
      RealVector point(mAlgorithm.dims());
      point <<= buf.samps(0, mAlgorithm.dims(), 0);
      bool result = mAlgorithm.update(id, point);
      if (result) return OK();
    }
    return addPoint(id, data);
  }

  MessageResult<void> deletePoint(string id)
  {
    return mAlgorithm.remove(id) ? OK() : Error(PointNotFound);
  }

  MessageResult<void> merge(SharedClientRef<const DataSetClient> datasetClient,
                            bool                           overwrite)
  {
    auto datasetClientPtr = datasetClient.get().lock();
    if (!datasetClientPtr) return Error(NoDataSet);
    auto srcDataSet = datasetClientPtr->getDataSet();
    if (srcDataSet.size() == 0) return Error(EmptyDataSet);
    if (srcDataSet.pointSize() != mAlgorithm.pointSize())
      return Error(WrongPointSize);
    auto       ids = srcDataSet.getIds();
    RealVector point(srcDataSet.pointSize());
    for (index i = 0; i < srcDataSet.size(); i++)
    {
      srcDataSet.get(ids(i), point);
      bool added = mAlgorithm.add(ids(i), point);
      if (!added && overwrite) mAlgorithm.update(ids(i), point);
    }
    return OK();
  }

  MessageResult<void>
  fromBuffer(InputBufferPtr data, bool transpose,
             SharedClientRef<const labelset::LabelSetClient> labels)
  {
    if (!data) return Error(NoBuffer);
    BufferAdaptor::ReadAccess buf(data.get());
    if (!buf.exists()) return Error(InvalidBuffer);
    auto bufView = transpose ? buf.allFrames() : buf.allFrames().transpose();
    if (auto labelsPtr = labels.get().lock())
    {
      auto& labelSet = labelsPtr->getLabelSet();
      if (labelSet.size() != bufView.rows())
      { return Error("Label set size needs to match the buffer size"); }
      mAlgorithm = DataSet(labelSet.getData().col(0),
                           FluidTensorView<const float, 2>(bufView));
    }
    else
    {
      algorithm::DataSetIdSequence seq("", 0, 0);
      FluidTensor<string, 1>       newIds(bufView.rows());
      seq.generate(newIds);
      mAlgorithm = DataSet(newIds, FluidTensorView<const float, 2>(bufView));
    }
    return OK();
  }

  MessageResult<void> toBuffer(BufferPtr data, bool transpose,
                               LabelSetClientRef labels)
  {
    if (!data) return Error(NoBuffer);
    BufferAdaptor::Access buf(data.get());
    if (!buf.exists()) return Error(InvalidBuffer);
    index  nFrames = transpose ? mAlgorithm.dims() : mAlgorithm.size();
    index  nChannels = transpose ? mAlgorithm.size() : mAlgorithm.dims();
    Result resizeResult = buf.resize(nFrames, nChannels, buf.sampleRate());
    if (!resizeResult.ok()) return Error(resizeResult.message());
    buf.allFrames() <<=
        transpose ? mAlgorithm.getData()
                  : FluidTensorView<const double, 2>(mAlgorithm.getData())
                        .transpose();
    auto labelsPtr = labels.get().lock();
    if (labelsPtr) labelsPtr->setLabelSet(getIdsLabelSet());
    return OK();
  }

  MessageResult<void> getIds(LabelSetClientRef dest)
  {
    auto destPtr = dest.get().lock();
    if (!destPtr) return Error(NoDataSet);
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

    InBufferCheck bufCheck(mAlgorithm.dims());

    if (!bufCheck.checkInputs(data.get()))
      return Error<FluidTensor<rt::string, 1>>(bufCheck.error());

    FluidTensor<const double, 1> point(
        BufferAdaptor::ReadAccess(data.get()).samps(0, mAlgorithm.dims(), 0));

    std::vector<index> indices(asUnsigned(mAlgorithm.size()));
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<double> distances(asUnsigned(mAlgorithm.size()));

    auto ds = mAlgorithm.getData();

    std::transform(
        indices.begin(), indices.end(), distances.begin(),
        [&point, &ds, this](index i) { return distance(point, ds.row(i)); });

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

    InBufferCheck bufCheck(mAlgorithm.dims());

    if (!bufCheck.checkInputs(data.get()))
      return Error<FluidTensor<double, 1>>(bufCheck.error());

    FluidTensor<const double, 1> point(
        BufferAdaptor::ReadAccess(data.get()).samps(0, mAlgorithm.dims(), 0));

    std::vector<index> indices(asUnsigned(mAlgorithm.size()));
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<double> distances(asUnsigned(mAlgorithm.size()));

    auto ds = mAlgorithm.getData();

    std::transform(
        indices.begin(), indices.end(), distances.begin(),
        [&point, &ds, this](index i) { return distance(point, ds.row(i)); });

    std::sort(indices.begin(), indices.end(), [&distances](index a, index b) {
      return distances[asUnsigned(a)] < distances[asUnsigned(b)];
    });

    FluidTensor<double, 1> labels(nNeighbours);

    std::transform(
        indices.begin(), indices.begin() + nNeighbours, labels.begin(),
        [&distances](index i) { return distances[asUnsigned(i)]; });

    return labels;
  }

  MessageResult<void> clear()
  {
    mAlgorithm = DataSet(0);
    return OK();
  }
  
  MessageResult<string> print()
  {
    return "DataSet " + std::string(get<kName>()) + ": " + mAlgorithm.print();
  }

  const DataSet getDataSet() const { return mAlgorithm; }
  void          setDataSet(DataSet ds) { mAlgorithm = ds; }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("addPoint", &DataSetClient::addPoint),
        makeMessage("getPoint", &DataSetClient::getPoint),
        makeMessage("setPoint", &DataSetClient::setPoint),
        makeMessage("updatePoint", &DataSetClient::updatePoint),
        makeMessage("deletePoint", &DataSetClient::deletePoint),
        makeMessage("merge", &DataSetClient::merge),
        makeMessage("dump", &DataSetClient::dump),
        makeMessage("load", &DataSetClient::load),
        makeMessage("print", &DataSetClient::print),
        makeMessage("size", &DataSetClient::size),
        makeMessage("cols", &DataSetClient::dims),
        makeMessage("clear", &DataSetClient::clear),
        makeMessage("write", &DataSetClient::write),
        makeMessage("read", &DataSetClient::read),
        makeMessage("fromBuffer", &DataSetClient::fromBuffer),
        makeMessage("toBuffer", &DataSetClient::toBuffer),
        makeMessage("getIds", &DataSetClient::getIds),
        makeMessage("kNearestDist", &DataSetClient::kNearestDist),
        makeMessage("kNearest", &DataSetClient::kNearest));
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
  
  double distance(FluidTensorView<const double, 1> point1, FluidTensorView<const double, 1> point2) const
  {    
    return std::transform_reduce(point1.begin(), point1.end(), point2.begin(), 0.0, std::plus{}, [](double v1, double v2){
      return (v1-v2) * (v1-v2);
    });
  };
};

} // namespace dataset

using DataSetClientRef = SharedClientRef<dataset::DataSetClient>;
using InputDataSetClientRef = SharedClientRef<const dataset::DataSetClient>;

using NRTThreadedDataSetClient =
    NRTThreadingAdaptor<typename DataSetClientRef::SharedType>;

} // namespace client
} // namespace fluid
