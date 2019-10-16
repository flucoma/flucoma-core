#pragma once

#include "data/FluidDataSet.hpp"
#include "FluidSharedInstanceAdaptor.hpp"
#include "../common/SharedClientUtils.hpp"
#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/MessageSet.hpp>
#include <clients/common/OfflineClient.hpp>
#include <clients/common/ParameterSet.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/Result.hpp>
#include <clients/nrt/FluidNRTClientWrapper.hpp>
#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>
#include <memory>
#include <unordered_map>
#include <string>

namespace fluid {
namespace client {

class ProviderTestClient : public FluidBaseClient,public OfflineIn, public OfflineOut
{

  enum { kName, kDummy };
  using string = std::string;

public:

  //external messages omitted

  struct Entry
  {
    intptr_t offset;
    intptr_t length;
  };

  FLUID_DECLARE_PARAMS(
    StringParam<Fixed<true>>("name", "Provider name"),
    LongParam("dummy","Checking Attrui Updates",0)
  );

  using ProviderDataSet = FluidDataSet<string, Entry, 1>;

  ProviderTestClient(ParamSetViewType &p):mParams(p), mTmp(1){}

  template <typename T>
  Result process(FluidContext&) { return {}; }

  std::string name() const { return get<kName>(); }

  MessageResult<void> addPoint(string label, intptr_t offset,intptr_t length)
  {
    mTmp.row(0) = Entry{offset,length};
    return mData.add(label,mTmp)
               ? MessageResult<void>{Result::Status::kOk}
               : MessageResult<void>{Result::Status::kError, "Label already exists"};
  }

  MessageResult<Entry> getPoint(string label) const
  {
      FluidTensor<Entry, 1> data(1);
      bool  result = mData.get(label, data);
      if(result)
        return {data.row(0)};
      else
        return {Result::Status::kError,"Couldn't retreive data"};
  }

  MessageResult<void> updatePoint(string label,intptr_t offset,intptr_t length)
  {
   mTmp.row(0) = Entry{offset,length};
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

  FLUID_DECLARE_MESSAGES(
    makeMessage("addPoint",    &ProviderTestClient::addPoint),
    makeMessage("updatePoint", &ProviderTestClient::updatePoint),
    makeMessage("deletePoint", &ProviderTestClient::deletePoint)
  );

private:
  mutable ProviderDataSet mData{1};
  FluidTensor<Entry,1> mTmp;
};
//this alias is what is used by subscribing clients, and is all that's needed to make a client a provider
using ProviderTestClientRef = SharedClientRef<ProviderTestClient>;
using NRTThreadedProviderTest = NRTThreadingAdaptor<typename ProviderTestClientRef::SharedType>;

} // namespace client
} // namespace fluid
