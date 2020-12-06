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


class BufScaleClient :    public FluidBaseClient,
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
    kInLow,
    kInHigh,
    kOutLow, 
    kOutHigh,
    kMap,
    kCurve,
    kClip
  };
  public:
  
  FLUID_DECLARE_PARAMS(InputBufferParam("source", "Source Buffer"),
                       LongParam("startFrame", "Source Offset", 0, Min(0)),
                       LongParam("numFrames", "Number of Frames", -1),
                       LongParam("startChan", "Start Channel", 0, Min(0)),
                       LongParam("numChans", "Number of Channels", -1),
                       BufferParam("destination","Destination Buffer"), 
                       FloatParam("inputLow","Input Low Range",0),
                       FloatParam("inputHigh","Input High Range",1),
                       FloatParam("outputLow","Output Low Range",0),
                       FloatParam("outputHigh","Output High Range",1),
                       EnumParam("map","Type of Mapping",0,"linlin", "lincurve", "curvelin"),
                       FloatParam("curvature", "Curvature of the Mapping", 0),
                       EnumParam("clipping","Optional Clipping",0,"Neither", "Minimum", "Maximum", "Both"));
  
  BufScaleClient(ParamSetViewType& p) : mParams(p) {}                   
  
  template <typename T>
  Result process(FluidContext&)
  {
    // retrieve the range requested and check it is valid
    index startFrame = get<kStartFrame>();
    index numFrames  = get<kNumFrames>();
    index startChan  = get<kStartChan>();
    index numChans   = get<kNumChans>();
    
    Result r = bufferRangeCheck(get<kSource>().get(), startFrame, numFrames, startChan, numChans);
    
    if(! r.ok())  return r;

    BufferAdaptor::ReadAccess source(get<kSource>().get());
    BufferAdaptor::Access     dest(get<kDest>().get());
    
    if (!dest.exists())
      return {Result::Status::kError, "Output buffer not found"};

    FluidTensor<double, 2> tmp(source.allFrames()(Slice(startChan,numChans),Slice(startFrame,numFrames)));
        
    //process
    double scale = (get<kOutHigh>()-get<kOutLow>())/(get<kInHigh>()-get<kInLow>());
    double offset = get<kOutLow>() - ( scale * get<kInLow>() ); 
    
    tmp.apply([&](double& x){
        //optional cliping
        if ((get<kClip>() & 1) && (x < get<kInLow>())) { //if the low clipping is on, and X  is below the low input
            x = get<kOutLow>();
        } else if ((get<kClip>() & 2) && (x > get<kInHigh>())) { //if the high clipping is on, and X is above the high input
            x = get<kOutHigh>();
        } else if ((get<kMap>() == 0) || (abs(get<kCurve>()) < 0.001)){ //if Map is linear, or the curve is linear enough (SC optimisation trick)
            x *= scale;
            x += offset;
        } else if (get<kMap>() == 1) { // if map is linExp (taken from SC's lincurve code but with added parentheses)
            double grow = exp(get<kCurve>());//this could be an instance constant updated when kCurve is changed to save computing it every time
            double a = (get<kOutHigh>() - get<kOutLow>()) / (1.0 - grow);
            double b = get<kOutLow>() + a;//this could be inlined 2 lines below
            double scaled = (x - get<kInLow>()) / (get<kInHigh>() - get<kInLow>());
            x = b - (a * pow(grow, scaled));
        } else { // otherwise it should be expLin (taken from SC's curvelin code but with added parentheses)
            assert(get<kMap>() == 2); //just checkin'
            double grow = exp(get<kCurve>()); //again this should be stored somewhere when curve changes
            double a = (get<kInHigh>() - get<kInLow>()) / (1.0 - grow);
            double b = get<kInLow>() + a; //this coulb be inldined below
            x = (log((b - x) / a) * (get<kOutHigh>() - get<kOutLow>()) / get<kCurve>()) + get<kOutLow>();
        }
    }); 

    //write back the processed data, resizing the dest buffer          
    r = dest.resize(numFrames,numChans,source.sampleRate());
    if(!r.ok()) return r;
    
    dest.allFrames() = tmp; 
    
//    for(index i = 0; i < numChans; ++i)
//       dest.samps(i) = tmp.row(i);
    
    return {};
  }                  
}; 

using NRTThreadedBufferScaleClient =
    NRTThreadingAdaptor<ClientWrapper<BufScaleClient>>;
}//client
}//fluid
