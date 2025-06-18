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

#include "../common/BufferAdaptor.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterTypes.hpp"

namespace fluid {
namespace client {
namespace bufthresh {

enum BufSelectParamIndex {
  kSource,
  kStartFrame,
  kNumFrames,
  kStartChan,
  kNumChans,
  kDest,
  kThresh
};

constexpr auto BufThreshParams =
    defineParameters(InputBufferParam("source", "Source Buffer"),
                     LongParam("startFrame", "Source Offset", 0, Min(0)),
                     LongParam("numFrames", "Number of Frames", -1),
                     LongParam("startChan", "Start Channel", 0, Min(0)),
                     LongParam("numChans", "Number of Channels", -1),
                     BufferParam("destination", "Destination Buffer"),
                     FloatParam("threshold", "Threshold", 0));

class BufThreshClient : public FluidBaseClient,
                        public OfflineIn,
                        public OfflineOut
{


public:
  using ParamDescType = decltype(BufThreshParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return BufThreshParams; }


  BufThreshClient(ParamSetViewType& p, FluidContext&) : mParams(p) {}

  template <typename T>
  Result process(FluidContext&)
  {
    // retrieve the range requested and check it is valid
    index startFrame = get<kStartFrame>();
    index numFrames = get<kNumFrames>();
    index startChan = get<kStartChan>();
    index numChans = get<kNumChans>();

    Result r = bufferRangeCheck(get<kSource>().get(), startFrame, numFrames,
                                startChan, numChans);

    if (!r.ok()) return r;

    BufferAdaptor::ReadAccess source(get<kSource>().get());
    BufferAdaptor::Access     dest(get<kDest>().get());

    if (!dest.exists())
      return {Result::Status::kError, "Output buffer not found"};

    FluidTensor<float, 2> tmp(numChans, numFrames);

    for (index i = 0; i < numChans; ++i)
      tmp.row(i) <<= source.samps(startFrame, numFrames, (i + startChan));

    // process
    double threshold = get<kThresh>();

    tmp.apply([&threshold](float& x) { x = x < threshold ? 0 : x; });

    // write back the processed data, resizing the dest buffer
    r = dest.resize(numFrames, numChans, source.sampleRate());

    if (!r.ok()) return r;

    for (index i = 0; i < numChans; ++i) dest.samps(i) <<= tmp.row(i);

    return {};
  }
};
} // namespace bufthresh

using NRTThreadedBufferThreshClient =
    NRTThreadingAdaptor<ClientWrapper<bufthresh::BufThreshClient>>;
} // namespace client
} // namespace fluid
