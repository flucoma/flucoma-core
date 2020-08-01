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

class BufFlattenClient : public FluidBaseClient, OfflineIn, OfflineOut
{

public:
  enum {
    kSource,
    kDest,
    kAxis
  };

  FLUID_DECLARE_PARAMS(InputBufferParam("source", "Source Buffer"),
                       BufferParam("destination", "Destination Buffer"),
                       EnumParam("axis","Axis",1, "Frames", "Channels"));

  BufFlattenClient(ParamSetViewType& p) : mParams{p} {}

  template <typename T>
  Result process(FluidContext&)
  {

    if (!get<kSource>().get()) { return {Result::Status::kError, "No source buffer "}; }
    if (!get<kDest>().get()) { return  {Result::Status::kError, "No destination buffer"}; }

    BufferAdaptor::ReadAccess source(get<kSource>().get());
    BufferAdaptor::Access destination(get<kDest>().get()); 

    if (!(source.exists() && source.valid()))
      return {Result::Status::kError, "Source Buffer Not Found or Invalid"};
      
    if (!destination.exists())
      return {Result::Status::kError,
              "Destination Buffer Not Found or Invalid"};  
                            
    auto resizeResult = destination.resize(source.numFrames()*source.numChans(), 1 ,source.sampleRate());
    
    if(!resizeResult.ok()) return resizeResult; 
    
    
    if(get<kAxis>() == 0)
      std::copy(source.allFrames().begin(),
                source.allFrames().end(),
                destination.allFrames().begin()); 
    else                                             
      std::copy(source.allFrames().transpose().begin(),
                source.allFrames().transpose().end(),
                destination.allFrames().begin()); 

    return {Result::Status::kOk};
  }
};

using NRTThreadedBufFlattenClient =
    NRTThreadingAdaptor<ClientWrapper<BufFlattenClient>>;

} // namespace client
} // namespace fluid
