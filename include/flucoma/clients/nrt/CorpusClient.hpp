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
#include "../../data/FluidDataset.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include <memory>
#include <string>
#include <unordered_map>

namespace fluid {
namespace client {
namespace corpus {

enum { kName };

constexpr auto CorpusParams =
    defineParameters(StringParam<Fixed<true>>("name", "Corpus name"));

class CorpusClient : public FluidBaseClient, public OfflineIn, public OfflineOut
{
  using string = std::string;
  using Buffer = typename BufferT::type;
  struct Entry
  {
    Buffer buffer;
    size_t offset{0};
    int    length{-1};
  };

public:
  using ParamDescType = decltype(CorpusParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return CorpusParams; }

  using CorpusDataSet = FluidDataSet<string, Entry, 1>;

  CorpusClient(ParamSetViewType& p, FluidContext&) : mParams(p), mTmp(1) {}

  template <typename T>
  Result process(FluidContext&)
  {
    return {};
  }

  std::string name() const { return get<kName>(); }

  MessageResult<void> addPoint(string label, Buffer buffer, size_t offset,
                               int length)
  {
    mTmp.row(0) = Entry{buffer, offset, length};
    return mCorpus.add(label, mTmp)
               ? MessageResult<void>{Result::Status::kOk}
               : MessageResult<void>{Result::Status::kError,
                                     "Label already in Corpus"};
  }

  MessageResult<std::tuple<Buffer, size_t, int>> getPoint(string label) const
  {
    FluidTensor<Entry, 1> data(1);
    bool                  result = mCorpus.get(label, data);
    if (result)
    {
      Entry e = data[0];
      return std::make_tuple(e.buffer, e.offset, e.length);
    }
    else
      return {Result::Status::kError, "Couldn't retreive data"};
  }

  MessageResult<void> updatePoint(string label, Buffer buffer, size_t offset,
                                  int length)
  {
    mTmp.row(0) = Entry{buffer, offset, length};
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


  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("addPoint", &CorpusClient::addPoint),
        makeMessage("updatePoint", &CorpusClient::updatePoint),
        makeMessage("deletePoint", &CorpusClient::deletePoint));
  }

private:
  mutable CorpusDataSet mCorpus{1};
  FluidTensor<Entry, 1> mTmp;
};
} // namespace corpus

using CorpusClientRef = SharedClientRef<CorpusClient>;
using NRTThreadedCorpus =
    NRTThreadingAdaptor<typename CorpusClientRef::SharedType>;

} // namespace client
} // namespace fluid
