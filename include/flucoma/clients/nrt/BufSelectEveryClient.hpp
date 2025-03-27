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
#include <algorithm> //std::iota
#include <vector>

namespace fluid {
namespace client {
namespace bufselectevery {

enum {
  kSource,
  kOffset,
  kNumFrames,
  kStartChan,
  kNumChans,
  kDest,
  kFrameHop,
  kChannelHop
};

constexpr auto BufSelectEveryParams =
    defineParameters(InputBufferParam("source", "Source Buffer"),
                     LongParam("startFrame", "Source Offset", 0, Min(0)),
                     LongParam("numFrames", "Source Number of Frames", -1),
                     LongParam("startChan", "Source Channel Offset", 0, Min(0)),
                     LongParam("numChans", "Source Number of Channels", -1),
                     BufferParam("destination", "Destination Buffer"),
                     LongParam("frameHop", "Frame Hop", 1),
                     LongParam("chanHop", "Channel Hop", 1));


class BufSelectEveryClient : public FluidBaseClient, OfflineIn, OfflineOut
{
public:
  using ParamDescType = decltype(BufSelectEveryParams);

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
    return BufSelectEveryParams;
  }


  BufSelectEveryClient(ParamSetViewType& p, FluidContext&) : mParams{p} {}

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


    index offset = get<kOffset>();
    index startChan = get<kStartChan>();

    if (offset >= source.numFrames())
      return {Result::Status::kError, "Start frame (", offset,
              ") out of range."};

    if (startChan >= source.numChans())
      return {Result::Status::kError, "Start channel ", startChan,
              " out of range."};

    index numFrames =
        get<kNumFrames>() < 0 ? source.numFrames() - offset : get<kNumFrames>();

    index numChans =
        get<kNumChans>() < 0 ? source.numChans() - startChan : get<kNumChans>();

    float framehop = get<kFrameHop>();
    float chanhop = get<kChannelHop>();

    numFrames = static_cast<index>(std::round(numFrames / framehop));
    numChans = static_cast<index>(std::round(numChans / chanhop));


    if (numChans <= 0 || numFrames <= 0)
      return {Result::Status::kError, "Zero length segment requested"};

    auto resizeResult =
        destination.resize(numFrames, numChans, source.sampleRate());

    if (!resizeResult.ok()) return resizeResult;

    std::vector<index> indices(asUnsigned(numFrames));
    std::vector<index> channels(asUnsigned(numChans));

    std::generate(
        indices.begin(), indices.end(),
        [framehop, n = offset - framehop]() mutable { return n += framehop; });

    std::generate(
        channels.begin(), channels.end(),
        [chanhop, n = startChan - chanhop]() mutable { return n += chanhop; });

    auto dest = destination.allFrames();
    auto src = source.allFrames();

    for (index c = 0; c < numChans; ++c)
      for (index i = 0; i < numFrames; ++i)
        dest(c, i) = src(channels[asUnsigned(c)], indices[asUnsigned(i)]);

    return {Result::Status::kOk};
  }
};
} // namespace bufselectevery

using NRTThreadingSelectEveryClient =
    NRTThreadingAdaptor<ClientWrapper<bufselectevery::BufSelectEveryClient>>;

} // namespace client
} // namespace fluid
