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

#include "ProviderTest.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/MessageSet.hpp"
#include "../common/OfflineClient.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../common/Result.hpp"
#include "data/FluidDataSet.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include <string>


namespace fluid {
namespace client {
namespace subscribertest {

enum { kProvider };

constexpr auto SubscriberTestParams =
    defineParameters(ProviderTestClientRef::makeParam("provider", "Provider"));

class SubscriberTestClient : public FluidBaseClient, OfflineIn, OfflineOut
{
public:
  using string = std::string;

  template <typename T>
  Result process(FluidContext&)
  {
    return {};
  }

  using ParamDescType = decltype(SubscriberTestParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto getParameterDescriptors()
  {
    return SubscriberTestParams;
  }

  SubscriberTestClient(ParamSetViewType& p) : mParams(p) {}

  MessageResult<std::tuple<std::string, intptr_t, intptr_t>>
  getFromProivder(string label)
  {

    auto corpusRef = get<kProvider>().get(); // returns weak_ptr to client

    if (auto corpus =
            corpusRef.lock()) // this is how weak_ptr works, don't blame me
    {
      MessageResult<providertest::ProviderTestClient::Entry> response =
          corpus->getPoint(label);

      if (response.ok())
      {
        auto data =
            static_cast<providertest::ProviderTestClient::Entry>(response);
        return std::make_tuple(label, data.offset, data.length);
      }
      else
        return {Result::Status::kError, response.message()};
    }
    return {Result::Status::kError, "Provider doesn't exist"};
  }

  MessageResult<std::tuple<std::string, intptr_t, intptr_t>>
  getFromProivderByMessage(ProviderTestClientRef wrappedProvider, string label)
  {
    std::weak_ptr<const providertest::ProviderTestClient> providerWeakPointer =
        wrappedProvider.get();

    if (auto providerPointer = providerWeakPointer.lock())
    {
      MessageResult<providertest::ProviderTestClient::Entry> response =
          providerPointer->getPoint(label);

      if (response.ok())
      {
        auto data =
            static_cast<providertest::ProviderTestClient::Entry>(response);
        return std::make_tuple(label, data.offset, data.length);
      }
      else
        return {Result::Status::kError, response.message()};
    }
    return {Result::Status::kError, "Provider doesn't exist"};
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("providerLookup", &SubscriberTestClient::getFromProivder),
        makeMessage("providerLookupFromMessage",
                    &SubscriberTestClient::getFromProivderByMessage));
  }
};
} // namespace subscribertest

using NRTThreadedSubscriberTest =
    NRTThreadingAdaptor<ClientWrapper<subscribertest::SubscriberTestClient>>;

} // namespace client
} // namespace fluid
