#pragma once

#include "ProviderTest.hpp"

#include "data/FluidDataset.hpp"

#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/MessageSet.hpp>
#include <clients/common/OfflineClient.hpp>
#include <clients/common/ParameterSet.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/Result.hpp>
#include <clients/nrt/FluidNRTClientWrapper.hpp>
#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>
#include <string>


namespace fluid {
namespace client {

class SubscriberTestClient : public FluidBaseClient, OfflineIn, OfflineOut
{
    enum { kProvider };
  
public:
  using string = std::string;

  template <typename T>
  Result process(FluidContext&) { return {}; }

  FLUID_DECLARE_PARAMS(
    ProviderTestClientRef::MakeParam("provider","Provider")
  );

  SubscriberTestClient(ParamSetViewType &p) : mParams(p){}

  MessageResult<std::tuple<std::string,intptr_t,intptr_t>> getFromProivder(string label) {
    
      auto corpusRef = get<kProvider>().get(); //returns weak_ptr to client

      if(auto corpus = corpusRef.lock()) //this is how weak_ptr works, don't blame me
      {
        MessageResult<ProviderTestClient::Entry> response = corpus->getPoint(label);
      
        if(response.ok())
        {
          auto data = static_cast<ProviderTestClient::Entry>(response);
          return std::make_tuple(label,data.offset,data.length); 
        }
        else
          return {Result::Status::kError,response.message()};
      } return {Result::Status::kError,"Provider doesn't exist"};
  }

  FLUID_DECLARE_MESSAGES(
    makeMessage("providerLookup",&SubscriberTestClient::getFromProivder)
  );

};

using NRTThreadedSubscriberTest = NRTThreadingAdaptor<ClientWrapper<SubscriberTestClient>>;

} // namespace client
} // namespace fluid

