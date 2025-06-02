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
#include "NRTClient.hpp"
#include "../common/SharedClientUtils.hpp"
#include "../../algorithms/public/DataSetIdSequence.hpp"

namespace fluid {
namespace client {

//Do a forward declaration here so that LabelSetClient can refer to a LabelSetClientRef 
namespace labelset {
  class LabelSetClient;
}

//Note that the shared type alias is declared after the Client implementation in most other cases 
using LabelSetClientRef = SharedClientRef<labelset::LabelSetClient>;
using InputLabelSetClientRef = SharedClientRef<const labelset::LabelSetClient>;

namespace labelset {

enum { kName };

constexpr auto LabelSetParams =
    defineParameters(StringParam<Fixed<true>>("name", "Name of the LabelSet"));

class LabelSetClient
    : public FluidBaseClient,
      OfflineIn,
      OfflineOut,
      public DataClient<FluidDataSet<std::string, std::string, 1>>
{
public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using LabelSet = FluidDataSet<string, string, 1>;
  using StringVector = FluidTensor<string, 1>;
  using StringMatrix = FluidTensor<string, 2>;

  template <typename T>
  Result process(FluidContext&)
  {
    return {};
  }

  using ParamDescType = decltype(LabelSetParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return LabelSetParams; }

  LabelSetClient(ParamSetViewType& p, FluidContext&) : mParams(p) {}

  MessageResult<void> addLabel(string id, string label)
  {

    if (id.empty()) return Error(EmptyId);
    if (label.empty()) return Error(EmptyLabel);
    if (mAlgorithm.dims() == 0) { mAlgorithm = LabelSet(1); }
    StringVector point = {label};
    return mAlgorithm.add(id, point) ? OK() : Error(DuplicateIdentifier);
  }

  MessageResult<string> getLabel(string id) const
  {
    if (id.empty()) return Error<string>(EmptyId);
    StringVector point(1);
    mAlgorithm.get(id, point);
    return point(0);
  }

  MessageResult<void> updateLabel(string id, string label)
  {
    if (id.empty()) return Error(EmptyId);
    if (label.empty()) return Error(EmptyLabel);
    StringVector point = {label};
    return mAlgorithm.update(id, point) ? OK() : Error(PointNotFound);
  }

  MessageResult<void> setLabel(string id, string label)
  {
    if (id.empty()) return Error(EmptyId);
    if (label.empty()) return Error(EmptyLabel);
    bool result = updateLabel(id, label).ok();
    if (result) return OK();
    return addLabel(id, label);
  }

  MessageResult<void> deleteLabel(string id)
  {
    return mAlgorithm.remove(id) ? OK() : Error(PointNotFound);
  }

  MessageResult<void> merge(LabelSetClientRef labelsetClient,
                            bool                           overwrite)
  {
    auto labelsetClientPtr = labelsetClient.get().lock();
    if (!labelsetClientPtr) return Error(NoLabelSet);
    auto srcLabelSet = labelsetClientPtr->getLabelSet();
    if (!labelsetClientPtr) return Error(NoLabelSet);
    auto       ids = srcLabelSet.getIds();
    StringVector point(1);
    for (index i = 0; i < srcLabelSet.size(); i++)
    {
      srcLabelSet.get(ids(i), point);
      bool added = mAlgorithm.add(ids(i), point);
      if (!added && overwrite) mAlgorithm.update(ids(i), point);
    }
    return OK();
  }

  MessageResult<void> clear()
  {
    mAlgorithm = LabelSet(1);
    return OK();
  }

  MessageResult<void> getIds(LabelSetClientRef dest)
  {
    auto destPtr = dest.get().lock();
    if (!destPtr) return Error(NoDataSet);
    destPtr->setLabelSet(getIdsLabelSet());
    return OK();
  }

  MessageResult<string> print() 
  { 
    return "LabelSet " + std::string(get<kName>()) + ": " + mAlgorithm.print();
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("addLabel", &LabelSetClient::addLabel),
        makeMessage("getLabel", &LabelSetClient::getLabel),
        makeMessage("deleteLabel", &LabelSetClient::deleteLabel),
        makeMessage("updateLabel", &LabelSetClient::updateLabel),
        makeMessage("setLabel", &LabelSetClient::setLabel),
        makeMessage("merge", &LabelSetClient::merge),
        makeMessage("dump", &LabelSetClient::dump),
        makeMessage("load", &LabelSetClient::load),
        makeMessage("print", &LabelSetClient::print),
        makeMessage("size", &LabelSetClient::size),
        makeMessage("cols", &LabelSetClient::dims),
        makeMessage("clear", &LabelSetClient::clear),
        makeMessage("write", &LabelSetClient::write),
        makeMessage("read", &LabelSetClient::read),
        makeMessage("getIds", &LabelSetClient::getIds));
  }

  const LabelSet getLabelSet() const { return mAlgorithm; }
  void           setLabelSet(LabelSet ls) { mAlgorithm = ls; }
  
private: 
  LabelSet getIdsLabelSet()
  {
    algorithm::DataSetIdSequence seq("", 0, 0);
    FluidTensor<string, 1>       newIds(mAlgorithm.size());
    FluidTensor<string, 2>       labels(mAlgorithm.size(), 1);
    labels.col(0) <<= mAlgorithm.getIds();
    seq.generate(newIds);
    return LabelSet(newIds, labels);
  }  
};
} // namespace labelset


using NRTThreadedLabelSetClient =
    NRTThreadingAdaptor<typename LabelSetClientRef::SharedType>;

} // namespace client
} // namespace fluid
