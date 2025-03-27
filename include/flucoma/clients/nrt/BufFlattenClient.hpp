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

namespace fluid {
namespace client {
namespace bufflatten {

enum { kSource, kOffset, kNumFrames, kStartChan, kNumChans, kDest, kAxis };

constexpr auto BufFlattenParams =
    defineParameters(InputBufferParam("source", "Source Buffer"),
                     LongParam("startFrame", "Source Offset", 0, Min(0)),
                     LongParam("numFrames", "Number of Frames", -1),
                     LongParam("startChan", "Start Channel", 0, Min(0)),
                     LongParam("numChans", "Number of Channels", -1),
                     BufferParam("destination", "Destination Buffer"),
                     EnumParam("axis", "Axis", 1, "Frames", "Channels"));

class BufFlattenClient : public FluidBaseClient, OfflineIn, OfflineOut
{

public:
  using ParamDescType = decltype(BufFlattenParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return BufFlattenParams; }


  BufFlattenClient(ParamSetViewType& p,FluidContext&) : mParams{p} {}

  template <typename T>
  Result process(FluidContext&)
  {

    if (!get<kSource>().get())
    { return {Result::Status::kError, "No source buffer "}; }
    if (!get<kDest>().get())
    { return {Result::Status::kError, "No destination buffer"}; }

    auto  srcptr = get<kSource>().get();
    index startFrame = get<kOffset>();
    index numFrames = get<kNumFrames>();
    index numChans = get<kNumChans>();
    index startChan = get<kStartChan>();

    Result rangecheck =
        bufferRangeCheck(srcptr, startFrame, numFrames, startChan, numChans);

    if (!rangecheck.ok()) return rangecheck;

    BufferAdaptor::ReadAccess source(srcptr);
    BufferAdaptor::Access     destination(get<kDest>().get());

    if (!(source.exists() && source.valid()))
      return {Result::Status::kError, "Source Buffer Not Found or Invalid"};

    if (!destination.exists())
      return {Result::Status::kError,
              "Destination Buffer Not Found or Invalid"};

    auto resizeResult =
        destination.resize(numFrames * numChans, 1, source.sampleRate());

    if (!resizeResult.ok()) return resizeResult;

    auto frames = source.allFrames();

    auto frameSel = frames(Slice(startChan, numChans),
                                     Slice(startFrame, numFrames)); 
    
    auto frameSelT = frameSel.transpose(); 
                                     
    auto destFrames = destination.allFrames();                                 

    if (get<kAxis>() == 0)
      std::copy(frameSel.begin(), frameSel.end(), destFrames.begin());
    else
      std::copy(frameSelT.begin(), frameSelT.end(), destFrames.begin());

    return {Result::Status::kOk};
  }
};
} // namespace bufflatten

using NRTThreadedBufFlattenClient =
    NRTThreadingAdaptor<ClientWrapper<bufflatten::BufFlattenClient>>;

} // namespace client
} // namespace fluid
