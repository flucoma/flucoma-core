/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
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
namespace bufshingle {

enum{ kSource, kOffset, kNumFrames, kStartChan, kNumChans, kDest, kSize };

constexpr auto BufShingleParams =
    defineParameters(InputBufferParam("source", "Source Buffer"),
                     LongParam("startFrame", "Source Offset", 0, Min(0)),
                     LongParam("numFrames", "Number of Frames", -1),
                     LongParam("startChan", "Start Channel", 0, Min(0)),
                     LongParam("numChans", "Number of Channels", -1),
                     BufferParam("destination", "Destination Buffer"),
                     LongParam("size","Shingle Size",2, Min(1)));

class BufShingleClient : public FluidBaseClient, OfflineIn, OfflineOut
{

public:
  using ParamDescType = decltype(BufShingleParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return BufShingleParams; }

  BufShingleClient(ParamSetViewType& p) : mParams{p} {}

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

    if(numFrames < get<kSize>())
      return {Result::Status::kError,
            "Fewer frames than shingle size"};
    
    index pointSize = numChans * get<kSize>();   
    
    ///this gets more complex if we have strides > 1          
    index numShingles = numFrames - get<kSize>() + 1;
               
    auto resizeResult =
        destination.resize(pointSize, numShingles, source.sampleRate());

    if (!resizeResult.ok()) return resizeResult;

    auto srcFrames = source.allFrames()(Slice(startChan, numChans),
                                     Slice(startFrame, numFrames));
    auto destFrames = destination.allFrames(); 

    for(index i = 0; i < numShingles; ++i) 
    {
      auto shingle = srcFrames(Slice(0,numChans),Slice(i,get<kSize>())).transpose(); 
      auto destRow = destFrames(i,Slice(0,pointSize)); 
      std::copy(shingle.begin(),shingle.end(), destRow.begin()); 
    }

    return {Result::Status::kOk};
  }
};
} // namespace bufshingle

using NRTThreadedBufShingleClient =
    NRTThreadingAdaptor<ClientWrapper<bufshingle::BufShingleClient>>;

} // namespace client
} // namespace fluid
