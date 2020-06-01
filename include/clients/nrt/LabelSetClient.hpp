#pragma once

#include "NRTClient.hpp"
#include "../common/SharedClientUtils.hpp"
#include "DataSetClient.hpp"

namespace fluid {
namespace client {

class LabelSetClient : public FluidBaseClient, OfflineIn, OfflineOut {
  enum { kName };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using LabelSet = FluidDataSet<string, string, 1>;
  using StringVector = FluidTensor<string, 1>;
  using StringMatrix = FluidTensor<string, 2>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS(StringParam<Fixed<true>>("name", "DataSet"));

  LabelSetClient(ParamSetViewType &p) :
      mParams(p), mLabelSet(1), mDataClient(mLabelSet) {}


  MessageResult<void> addLabel(string id, string label) {
    if(id.empty()) return Error(EmptyId);
    if(label.empty()) return Error(EmptyLabel);
    StringVector point = {label};
    return mLabelSet.add(id, point) ? OK() : Error(DuplicateLabel);
  }

  MessageResult<string> getLabel(string id) const {
    if(id.empty()) return Error<string>(EmptyId);
    StringVector point(1);
    mLabelSet.get(id, point);
    return  point(0);
  }

  MessageResult<void> updateLabel(string id, string label) {
    if(id.empty()) return Error(EmptyId);
    if(label.empty()) return Error(EmptyLabel);
    StringVector point = {label};
    return mLabelSet.update(id, point) ? OK() : Error(PointNotFound);
  }

  MessageResult<void> deleteLabel(string id) {
    return mLabelSet.remove(id) ? OK() : Error(PointNotFound);
  }

  MessageResult<void> clear() {
    mLabelSet = LabelSet(1);
    return OK();
  }

 MessageResult<string> print() {return mLabelSet.print();}

 MessageResult<index> size() { return mDataClient.size(); }
 MessageResult<index> cols() { return mDataClient.dims(); }
 MessageResult<void> write(string fn) {return mDataClient.write(fn);}
 MessageResult<void> read(string fn) {return mDataClient.read(fn);}
 MessageResult<string> dump() { return mDataClient.dump();}
 MessageResult<void> load(string  s) { return mDataClient.load(s);}


  FLUID_DECLARE_MESSAGES(
      makeMessage("addLabel", &LabelSetClient::addLabel),
      makeMessage("getLabel", &LabelSetClient::getLabel),
      makeMessage("deleteLabel", &LabelSetClient::deleteLabel),
      makeMessage("dump", &LabelSetClient::dump),
      makeMessage("load", &LabelSetClient::load),
      makeMessage("print", &LabelSetClient::print),
      makeMessage("size", &LabelSetClient::size),
      makeMessage("cols", &LabelSetClient::cols),
      makeMessage("clear", &LabelSetClient::clear),
      makeMessage("write", &LabelSetClient::write),
      makeMessage("read", &LabelSetClient::read)
  );

  const LabelSet getLabelSet() const { return mLabelSet; }
  void setLabelSet(LabelSet ls) {mLabelSet = ls; }

private:
  LabelSet mLabelSet{1};
  DataClient<LabelSet> mDataClient;
};
using LabelSetClientRef = SharedClientRef<LabelSetClient>;
using NRTThreadedLabelSetClient =
    NRTThreadingAdaptor<typename LabelSetClientRef::SharedType>;

} // namespace client
} // namespace fluid
