#pragma once

#include "NRTClient.hpp"
#include "../common/SharedClientUtils.hpp"
#include "DataSetClient.hpp"

namespace fluid {
namespace client {

class LabelSetClient : public FluidBaseClient, OfflineIn, OfflineOut,
public DataClient<FluidDataSet<std::string, std::string, 1>> {
  enum { kName };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using LabelSet = FluidDataSet<string, string, 1>;
  using StringVector = FluidTensor<string, 1>;
  using StringMatrix = FluidTensor<string, 2>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS(StringParam<Fixed<true>>("name", "Name of the LabelSet"));

  LabelSetClient(ParamSetViewType &p) : mParams(p) {}

  MessageResult<void> addLabel(string id, string label) {

    if(id.empty()) return Error(EmptyId);
    if(label.empty()) return Error(EmptyLabel);
    if(mAlgorithm.dims()==0) {mAlgorithm= LabelSet(1);}
    StringVector point = {label};
    return mAlgorithm.add(id, point) ? OK() : Error(DuplicateLabel);
  }

  MessageResult<string> getLabel(string id) const {
    if(id.empty()) return Error<string>(EmptyId);
    StringVector point(1);
    mAlgorithm.get(id, point);
    return  point(0);
  }

  MessageResult<void> updateLabel(string id, string label) {
    if(id.empty()) return Error(EmptyId);
    if(label.empty()) return Error(EmptyLabel);
    StringVector point = {label};
    return mAlgorithm.update(id, point) ? OK() : Error(PointNotFound);
  }

  MessageResult<void> deleteLabel(string id) {
    return mAlgorithm.remove(id) ? OK() : Error(PointNotFound);
  }

  MessageResult<void> clear() {
    mAlgorithm = LabelSet(1);
    return OK();
  }

 MessageResult<string> print() {return mAlgorithm.print();}


  FLUID_DECLARE_MESSAGES(
      makeMessage("addLabel", &LabelSetClient::addLabel),
      makeMessage("getLabel", &LabelSetClient::getLabel),
      makeMessage("deleteLabel", &LabelSetClient::deleteLabel),
      makeMessage("dump", &LabelSetClient::dump),
      makeMessage("load", &LabelSetClient::load),
      makeMessage("print", &LabelSetClient::print),
      makeMessage("size", &LabelSetClient::size),
      makeMessage("cols", &LabelSetClient::dims),
      makeMessage("clear", &LabelSetClient::clear),
      makeMessage("write", &LabelSetClient::write),
      makeMessage("read", &LabelSetClient::read)
  );

  const LabelSet getLabelSet() const { return mAlgorithm; }
  void setLabelSet(LabelSet ls) {mAlgorithm = ls; }

};
using LabelSetClientRef = SharedClientRef<LabelSetClient>;
using NRTThreadedLabelSetClient =
    NRTThreadingAdaptor<typename LabelSetClientRef::SharedType>;

} // namespace client
} // namespace fluid
