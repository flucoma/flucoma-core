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

#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/OfflineClient.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../common/Result.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include <numeric> //std::iota
#include <vector>

namespace fluid {
namespace client {
namespace bufselect {

static constexpr std::initializer_list<index> SelectionDefaults = {-1};

enum { kSource, kDest, kIndices, kChannels };

constexpr auto BufSelectParams =
    defineParameters(InputBufferParam("source", "Source Buffer"),
                     BufferParam("destination", "Destination Buffer"),
                     LongArrayParam("indices", "Indices", SelectionDefaults),
                     LongArrayParam("channels", "Channels", SelectionDefaults));

class BufSelectClient : public FluidBaseClient, OfflineIn, OfflineOut
{

public:
  using ParamDescType = decltype(BufSelectParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return BufSelectParams; }


  BufSelectClient(ParamSetViewType& p, FluidContext&) : mParams{p} {}

  template <typename T>
  Result process(FluidContext&)
  {

    if (!get<kSource>().get())
    { return {Result::Status::kError, "No source buffer "}; }
    if (!get<kDest>().get())
    { return {Result::Status::kError, "No destination buffer"}; }

    BufferAdaptor::ReadAccess source(get<kSource>().get());
    BufferAdaptor::Access     destination(get<kDest>().get());

    if (!(source.exists() && source.valid()))
      return {Result::Status::kError, "Source Buffer Not Found or Invalid"};

    if (!destination.exists())
      return {Result::Status::kError,
              "Destination Buffer Not Found or Invalid"};

    FluidTensorView<index, 1> indexMap = get<kIndices>();
    FluidTensorView<index, 1> channelMap = get<kChannels>();

    if (indexMap.size() < 1)
      return {Result::Status::kError, "No indices supplied"};

    if (channelMap.size() < 1)
      return {Result::Status::kError, "No channels supplied"};

    bool allFrames = indexMap.size() == 1 && indexMap(0) == -1;
    bool allChannels = channelMap.size() == 1 && channelMap(0) == -1;


    if (!allFrames)
    {
      auto maxIndex = *std::max_element(indexMap.begin(), indexMap.end());
      if (maxIndex > source.numFrames() - 1)
        return {Result::Status::kError, "Index out of range(", maxIndex, ')'};

      auto minIndex = *std::min_element(indexMap.begin(), indexMap.end());
      if (minIndex < 0)
        return {Result::Status::kError, "Index out of range(", minIndex, ')'};
    }

    if (!allChannels)
    {
      auto maxChannel = *std::max_element(channelMap.begin(), channelMap.end());
      if (maxChannel > source.numChans() - 1)
        return {Result::Status::kError, "Channel out of range(", maxChannel,
                ')'};

      auto minChannel = *std::min_element(channelMap.begin(), channelMap.end());
      if (minChannel < 0)
        return {Result::Status::kError, "Channel out of range(", minChannel,
                ')'};
    }

    index numFrames = allFrames ? source.numFrames() : indexMap.size();
    index numChans = allChannels ? source.numChans() : channelMap.size();

    auto resizeResult =
        destination.resize(numFrames, numChans, source.sampleRate());

    if (!resizeResult.ok()) return resizeResult;

    auto indices = FluidTensor<index, 1>(numFrames);
    auto channels = FluidTensor<index, 1>(numChans);

    if (allFrames)
      std::iota(indices.begin(), indices.end(), 0);
    else
      indices <<= indexMap;

    if (allChannels)
      std::iota(channels.begin(), channels.end(), 0);
    else
      channels <<= channelMap;

    auto dest = destination.allFrames();
    auto src = source.allFrames();

    for (index c = 0; c < numChans; ++c)
      for (index i = 0; i < numFrames; ++i)
        dest(c, i) = src(channels[c], indices[i]);

    return {Result::Status::kOk};
  }
};
} // namespace bufselect

using NRTThreadingSelectClient =
    NRTThreadingAdaptor<ClientWrapper<bufselect::BufSelectClient>>;

} // namespace client
} // namespace fluid
