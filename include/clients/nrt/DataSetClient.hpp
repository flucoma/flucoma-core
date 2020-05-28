#pragma once
#include "../common/SharedClientUtils.hpp"
#include "NRTClient.hpp"
#include "data/FluidDataSet.hpp"
#include "data/FluidJSON.hpp"
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

namespace fluid {
namespace client {

class DataSetClient : public FluidBaseClient, OfflineIn, OfflineOut {
  enum { kName };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using DataSet = FluidDataSet<string, double, 1>;
  using StringVector = FluidTensor<string, 1>;
  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS(StringParam<Fixed<true>>("name", "DataSet"));

  DataSetClient(ParamSetViewType &p) : mParams(p), mDataSet(0) {}

  MessageResult<void> addPoint(string id, BufferPtr data) {
    if (!data)
      return Error(NoBuffer);
    BufferAdaptor::Access buf(data.get());
    if (!buf.exists())
      return Error(InvalidBuffer);
    if (buf.numFrames() == 0)
      return Error(EmptyBuffer);
    if (mDataSet.size() == 0) {
      if (mDataSet.pointSize() != buf.numFrames())
        mDataSet = DataSet(buf.numFrames());
      mDims = mDataSet.pointSize();
    } else if (buf.numFrames() != mDims)
      return Error(WrongPointSize);
    RealVector point(mDims);
    point = buf.samps(0, mDims, 0);
    return mDataSet.add(id, point) ? OK() : Error(DuplicateLabel);
  }

  MessageResult<void> getPoint(string id, BufferPtr data) const {
    if (!data)
      return Error(NoBuffer);
    BufferAdaptor::Access buf(data.get());
    if (!buf.exists())
      return Error(InvalidBuffer);
    Result resizeResult = buf.resize(mDims, 1, buf.sampleRate());
    if (!resizeResult.ok())
      return {resizeResult.status(), resizeResult.message()};
    RealVector point(mDims);
    point = buf.samps(0, mDims, 0);
    bool result = mDataSet.get(id, point);
    if (result) {
      buf.samps(0, mDims, 0) = point;
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
    if (buf.numFrames() < mDims)
      return Error(WrongPointSize);
    RealVector point(mDims);
    point = buf.samps(0, mDims, 0);
    return mDataSet.update(id, point) ? OK() : Error(PointNotFound);
  }

  MessageResult<void> deletePoint(string id) {
    return mDataSet.remove(id) ? OK() : Error(PointNotFound);
  }

  MessageResult<index> size() { return mDataSet.size(); }
  MessageResult<index> cols() { return mDataSet.pointSize(); }

  MessageResult<void> clear() {
    mDataSet = DataSet(0);
    return {};
  }

  MessageResult<void> write(string fileName) {
    auto file = JSONFile(fileName, "w");
    file.write(mDataSet);
    return file.ok() ? OK() : Error(file.error());
  }

  MessageResult<void> read(string fileName) {
    auto file = JSONFile(fileName, "r");
    nlohmann::json j = file.read();
    if(!file.ok())
    {
      return Error(file.error());
    }
    else
    {
      mDataSet = j.get<DataSet>();
    }
    return OK();
  }

  MessageResult<string> print() {
    using namespace std;
    if (mDataSet.size() == 0)
      return {"Empty dataset"};
    else
      return mDataSet.print();
  }

  MessageResult<string> dump() {
    nlohmann::json j = mDataSet;
    return j.dump();
  }

  MessageResult<void> load(std::string s) {
    using namespace std;
    using namespace nlohmann;
    json j = json::parse(s, nullptr, false);
    if (j.is_discarded())
    {
      return Error("Parse error");
    }
    else{
      mDataSet = j.get<DataSet>();
      mDims = mDataSet.pointSize();
      return OK();
    }
  }

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
                         makeMessage("cols", &DataSetClient::cols),
                         makeMessage("clear", &DataSetClient::clear),
                         makeMessage("write", &DataSetClient::write),
                         makeMessage("read", &DataSetClient::read));

  const DataSet getDataSet() const { return mDataSet; }

  void setDataSet(DataSet ds) {
    mDataSet = ds;
    mDims = ds.pointSize();
  }

private:
  DataSet mDataSet;
  index mDims{0};
};
using DataSetClientRef = SharedClientRef<DataSetClient>;
using NRTThreadedDataSetClient =
    NRTThreadingAdaptor<typename DataSetClientRef::SharedType>;

} // namespace client
} // namespace fluid
