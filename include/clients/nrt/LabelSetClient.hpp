/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
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

namespace fluid {
namespace client {
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

  LabelSetClient(ParamSetViewType& p) : mParams(p) {}

  MessageResult<void> addLabel(string id, string label)
  {

    if (id.empty()) return Error(EmptyId);
    if (label.empty()) return Error(EmptyLabel);
    if (mAlgorithm.dims() == 0) { mAlgorithm = LabelSet(1); }
    StringVector point = {label};
    return mAlgorithm.add(id, point) ? OK() : Error(DuplicateLabel);
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

  MessageResult<void> deleteLabel(string id)
  {
    return mAlgorithm.remove(id) ? OK() : Error(PointNotFound);
  }

  MessageResult<void> clear()
  {
    mAlgorithm = LabelSet(1);
    return OK();
  }

  MessageResult<string> print() { return mAlgorithm.print(); }

  static auto getMessageDescriptors()
  {
    return defineMessages(
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
        makeMessage("read", &LabelSetClient::read));
  }

  const LabelSet getLabelSet() const { return mAlgorithm; }
  void           setLabelSet(LabelSet ls) { mAlgorithm = ls; }
};
} // namespace labelset

using LabelSetClientRef = SharedClientRef<labelset::LabelSetClient>;
using NRTThreadedLabelSetClient =
    NRTThreadingAdaptor<typename LabelSetClientRef::SharedType>;

} // namespace client
} // namespace fluid
