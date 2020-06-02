#pragma once
#include "../common/SharedClientUtils.hpp"
#include "DataClient.hpp"
#include "NRTClient.hpp"
#include "data/FluidDataSet.hpp"
#include <sstream>
#include <string>

namespace fluid {
namespace client {

class DataSetClient : public FluidBaseClient, OfflineIn, OfflineOut,
  public DataClient<FluidDataSet<std::string, double, 1>> {
  enum { kName };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using DataSet = FluidDataSet<string, double, 1>;

  template <typename T> Result process(FluidContext &) { return {}; }
  FLUID_DECLARE_PARAMS(StringParam<Fixed<true>>("name", "DataSet"));

  DataSetClient(ParamSetViewType &p)
      : DataClient(mDataSet), mParams(p), mDataSet(0) {}

  MessageResult<void> addPoint(string id, BufferPtr data) {
    if (!data)
      return Error(NoBuffer);
    BufferAdaptor::Access buf(data.get());
    if (!buf.exists())
      return Error(InvalidBuffer);
    if (buf.numFrames() == 0)
      return Error(EmptyBuffer);
    if (mDataSet.size() == 0) {
      if (mDataSet.dims() != buf.numFrames())
        mDataSet = DataSet(buf.numFrames());
    } else if (buf.numFrames() != mDataSet.dims())
      return Error(WrongPointSize);
    RealVector point(mDataSet.dims());
    point = buf.samps(0, mDataSet.dims(), 0);
    return mDataSet.add(id, point) ? OK() : Error(DuplicateLabel);
  }

  MessageResult<void> getPoint(string id, BufferPtr data) const {
    if (!data)
      return Error(NoBuffer);
    BufferAdaptor::Access buf(data.get());
    if (!buf.exists())
      return Error(InvalidBuffer);
    Result resizeResult = buf.resize(mDataSet.dims(), 1, buf.sampleRate());
    if (!resizeResult.ok())
      return {resizeResult.status(), resizeResult.message()};
    RealVector point(mDataSet.dims());
    point = buf.samps(0, mDataSet.dims(), 0);
    bool result = mDataSet.get(id, point);
    if (result) {
      buf.samps(0, mDataSet.dims(), 0) = point;
      return OK();
    } else {
      return Error(PointNotFound);
    }
  }

  MessageResult<void> updatePoint(string id, BufferPtr data) {
    if (!data)
      return Error(NoBuffer);
    BufferAdaptor::Access buf(data.get());
    if (!buf.exists())
      return Error(InvalidBuffer);
    if (buf.numFrames() < mDataSet.dims())
      return Error(WrongPointSize);
    RealVector point(mDataSet.dims());
    point = buf.samps(0, mDataSet.dims(), 0);
    return mDataSet.update(id, point) ? OK() : Error(PointNotFound);
  }

  MessageResult<void> deletePoint(string id) {
    return mDataSet.remove(id) ? OK() : Error(PointNotFound);
  }

  MessageResult<void> clear() {mDataSet = DataSet(0); return OK();}
  MessageResult<string> print() {return mDataSet.print();}
  const DataSet getDataSet() const { return mDataSet; }
  void setDataSet(DataSet ds) {mDataSet = ds;}

  FLUID_DECLARE_MESSAGES(makeMessage("addPoint", &DataSetClient::addPoint),
                         makeMessage("getPoint", &DataSetClient::getPoint),
                         makeMessage("updatePoint",
                                     &DataSetClient::updatePoint),
                         makeMessage("deletePoint",
                                     &DataSetClient::deletePoint),
                         makeMessage("dump", &DataSetClient::dump),
                         makeMessage("load", &DataSetClient::load),
                         makeMessage("print", &DataSetClient::print),
                         makeMessage("size", &DataSetClient::size),
                         makeMessage("cols", &DataSetClient::dims),
                         makeMessage("clear", &DataSetClient::clear),
                         makeMessage("write", &DataSetClient::write),
                         makeMessage("read", &DataSetClient::read));
private:
  DataSet mDataSet;
};
using DataSetClientRef = SharedClientRef<DataSetClient>;
using NRTThreadedDataSetClient =
    NRTThreadingAdaptor<typename DataSetClientRef::SharedType>;
} // namespace client
} // namespace fluid
