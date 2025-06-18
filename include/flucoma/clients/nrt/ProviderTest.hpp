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

#include "FluidSharedInstanceAdaptor.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/MessageSet.hpp"
#include "../common/OfflineClient.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../common/Result.hpp"
#include "../common/SharedClientUtils.hpp"
#include "../../data/FluidDataSet.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include <memory>
#include <string>
#include <unordered_map>

namespace fluid {
namespace client {
namespace providertest {

enum { kName, kDummy };

constexpr auto ProviderTestParams =
    defineParameters(StringParam<Fixed<true>>("name", "Provider name"),
                     LongParam("dummy", "Checking Attrui Updates", 0));

class ProviderTestClient : public FluidBaseClient,
                           public OfflineIn,
                           public OfflineOut
{
  using string = std::string;

public:
  // external messages omitted

  struct Entry
  {
    intptr_t offset;
    intptr_t length;
  };

  using ParamDescType = decltype(ProviderTestParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors()
  {
    return ProviderTestParams;
  }

  using ProviderDataSet = FluidDataSet<string, Entry, 1>;

  ProviderTestClient(ParamSetViewType& p) : mParams(p), mTmp(1) {}

  template <typename T>
  Result process(FluidContext&)
  {
    return {};
  }

  std::string name() const { return get<kName>(); }

  MessageResult<void> addPoint(string label, intptr_t offset, intptr_t length)
  {
    mTmp.row(0) = Entry{offset, length};
    return mData.add(label, mTmp) ? MessageResult<void>{Result::Status::kOk}
                                  : MessageResult<void>{Result::Status::kError,
                                                        "Label already exists"};
  }

  MessageResult<Entry> getPoint(string label) const
  {
    FluidTensor<Entry, 1> data(1);
    bool                  result = mData.get(label, data);
    if (result)
      return {data.row(0)};
    else
      return {Result::Status::kError, "Couldn't retreive data"};
  }

  MessageResult<void> updatePoint(string label, intptr_t offset,
                                  intptr_t length)
  {
    mTmp.row(0) = Entry{offset, length};
    return mData.update(label, mTmp)
               ? MessageResult<void>{Result::Status::kOk}
               : MessageResult<void>{Result::Status::kError, "Point not found"};
  }

  MessageResult<void> deletePoint(string label)
  {
    return mData.remove(label)
               ? MessageResult<void>{Result::Status::kOk}
               : MessageResult<void>{Result::Status::kError, "Point not found"};
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("addPoint", &ProviderTestClient::addPoint),
        makeMessage("updatePoint", &ProviderTestClient::updatePoint),
        makeMessage("deletePoint", &ProviderTestClient::deletePoint));
  }

private:
  mutable ProviderDataSet mData{1};
  FluidTensor<Entry, 1>   mTmp;
};
} // namespace providertest

// this alias is what is used by subscribing clients, and is all that's needed
// to make a client a provider
using ProviderTestClientRef = SharedClientRef<providertest::ProviderTestClient>;
using NRTThreadedProviderTest =
    NRTThreadingAdaptor<typename ProviderTestClientRef::SharedType>;

} // namespace client
} // namespace fluid
