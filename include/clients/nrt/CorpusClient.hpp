#pragma once

#include "data/FluidDataset.hpp"
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


class CorpusClient : public FluidBaseClient,public OfflineIn, public OfflineOut
{
  enum { kName };
  
  using string = std::string;
  using Buffer = typename BufferT::type;
  struct Entry
  {
    Buffer   buffer;
    size_t offset{0};
    int length{-1};
  };
public:
  
  FLUID_DECLARE_PARAMS(
    StringParam<Fixed<true>>("name", "Corpus name")
  );
  
  using CorpusDataSet = FluidDataset<Entry,string, 1>;

  CorpusClient(ParamSetViewType &p):mParams(p), mTmp(1){}

  template <typename T>
  Result process(FluidContext&) { return {}; }
  
  std::string name() const { return get<kName>(); }
  
  MessageResult<void> addPoint(string label, Buffer buffer, size_t offset,int length)
  {
    mTmp.row(0) = Entry{buffer, offset,length};
    return mCorpus.add(label,mTmp)
               ? MessageResult<void>{Result::Status::kOk}
               : MessageResult<void>{Result::Status::kError, "Label already in Corpus"};
  }

  MessageResult<std::tuple<Buffer,size_t,int>> getPoint(string label) const
  {
      FluidTensor<Entry, 1> data(1);
      bool  result = mCorpus.get(label, data);
      if(result)
      {
        Entry e = data[0];
        return std::make_tuple(e.buffer,e.offset,e.length);
      }
      else
        return {Result::Status::kError,"Couldn't retreive data"}; 
  }

  MessageResult<void> updatePoint(string label,Buffer buffer, size_t offset,int length)
  {
    mTmp.row(0) = Entry{buffer, offset,length};
    return mCorpus.update(label, mTmp)
               ? MessageResult<void>{Result::Status::kOk}
               : MessageResult<void>{Result::Status::kError, "Point not found"};
  }

  MessageResult<void> deletePoint(string label)
  {
    return mCorpus.remove(label)
               ? MessageResult<void>{Result::Status::kOk}
               : MessageResult<void>{Result::Status::kError, "Point not found"};
  }

  FLUID_DECLARE_MESSAGES(
    makeMessage("addPoint",    &CorpusClient::addPoint),
    makeMessage("updatePoint", &CorpusClient::updatePoint),
    makeMessage("deletePoint", &CorpusClient::deletePoint)
  );
  
private:
  mutable CorpusDataSet mCorpus{1};
  FluidTensor<Entry,1> mTmp;
};

using CorpusClientRef = SharedClientRef<CorpusClient>;
using NRTThreadedCorpus = NRTThreadingAdaptor<typename CorpusClientRef::SharedType>;

} // namespace client
} // namespace fluid
