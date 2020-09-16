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
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterTypes.hpp"

namespace fluid {
namespace client{

static constexpr std::initializer_list<index> BufSelectionDefaults = {-1};


class BufThreshClient :    public FluidBaseClient,
                          public OfflineIn,
                          public OfflineOut
{

  enum BufSelectParamIndex {
    kSource,
    kStartFrame,
    kNumFrames,
    kStartChan,
    kNumChans,
    kDest, 
    kThresh
  };
  public:
  
  FLUID_DECLARE_PARAMS(InputBufferParam("source", "Source Buffer"),
                       LongParam("startFrame", "Source Offset", 0, Min(0)),
                       LongParam("numFrames", "Number of Frames", -1),
                       LongParam("startChan", "Start Channel", 0, Min(0)),
                       LongParam("numChans", "Number of Channels", -1),
                       BufferParam("destination","Destination Buffer"), 
                       FloatParam("threshold","Threshold",0)); 
  
  BufThreshClient(ParamSetViewType& p) : mParams(p) {}                   
  
  template <typename T>
  Result process(FluidContext&)
  {
    
    if (!get<kSource>().get())
      return {Result::Status::kError, "No input buffer supplied"};

    if (!get<kDest>().get())
      return {Result::Status::kError, "No output buffer supplied"};

    BufferAdaptor::ReadAccess source(get<kSource>().get());
    BufferAdaptor::Access     dest(get<kDest>().get());

    if (!source.exists())
      return {Result::Status::kError, "Input buffer not found"};

    if (!source.valid())
      return {Result::Status::kError, "Can't access input buffer"};

    if (!dest.exists())
      return {Result::Status::kError, "Output buffer not found"};
    
    // retrieve the range requested and check it is valid
    index startFrame = get<kStartFrame>(); 
    // index numFrames  = get<kNumFrames>(); 
    index startChan  = get<kStartChan>();
    // index numChans   = get<kNumChans>();
    
    // Result r = bufferRangeCheck(source, startFrame, numFrames, startChan, numChans);
    // if(! r.ok())  return r;
    index numFrames = get<kNumFrames>() == -1
                          ? (source.numFrames() - get<kStartFrame>())
                          : get<kNumFrames>();
    index numChans = get<kNumChans>() == -1
                            ? (source.numChans() - get<kStartChan>())
                            : get<kNumChans>();

    if (get<kStartFrame>() + numFrames > source.numFrames())
      return {Result::Status::kError, "Start frame + num frames (",
              get<kStartFrame>() + get<kNumFrames>(), ") out of range."};

    if (get<kStartChan>() + numChans > source.numChans())
      return {Result::Status::kError, "Start channel ", get<kStartChan>(),
              " out of range."};

    if (numChans <= 0 || numFrames <= 0)
      return {Result::Status::kError, "Zero length segment requested"};
    // import the data to be processed in a temp Tensor
    
    FluidTensor<float, 2> tmp(numChans,numFrames);
  
    for(index i = 0; i < numChans; ++i)
      tmp.row(i) = source.samps(startFrame, numFrames, (i + startChan));
    
    //process  
    double threshold = get<kThresh>(); 

    tmp.apply([&threshold](float& x){
        x = x < threshold ? 0 : x;  
    }); 
    
    //write back the processed data, resizing the dest buffer  
    dest.resize(numFrames,numChans,source.sampleRate());
    
    for(index i = 0; i < numChans; ++i)
       dest.samps(i) = tmp.row(i);
    
    return {};
  }                  
}; 

using NRTThreadedBufferThreshClient =
    NRTThreadingAdaptor<ClientWrapper<BufThreshClient>>;
}//client
}//fluid
