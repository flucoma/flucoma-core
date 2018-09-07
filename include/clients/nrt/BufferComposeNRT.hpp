#pragma once

#include "clients/common/FluidParams.hpp"
#include "data/FluidTensor.hpp"
#include "data/FluidBuffers.hpp"

#include <string>
#include <sstream> //for ostringstream
#include <utility> //for std make_pair
#include <unordered_set>
#include <vector> //for containers of params, and for checking things

namespace fluid {
  namespace buf{
    
    /**
     Integration class for doing NMF filtering and resynthesis
     **/
    class BufferComposeClient
    {
      using desc_type = parameter::Descriptor;
      using param_type = parameter::Instance;
    public:
      
      struct ProcessModel
      {
        size_t frames[2];
        size_t offset[2];
        size_t channels[2];
        size_t channelOffset[2];
        parameter::BufferAdaptor* src1 = 0;
        parameter::BufferAdaptor* src2 = 0;
        parameter::BufferAdaptor* dst = 0;
        double gain[2];
      };
      
      static const std::vector<parameter::Descriptor>& getParamDescriptors()
      {
        static std::vector<desc_type> params;
        if(params.empty())
        {
          params.emplace_back(desc_type{"src","First Source Buffer", parameter::Type::Buffer});
          params.back().setInstantiation(true);
          
          params.emplace_back(desc_type{"offsetframes1","Source 1 Offset", parameter::Type::Long});
          params.back().setInstantiation(true).setMin(0).setDefault(0);

          params.emplace_back(desc_type{"numframes1","Source 1 Frames", parameter::Type::Long});
          params.back().setInstantiation(true).setMin(-1).setDefault(-1);
          
          params.emplace_back(desc_type{"offsetchans1","Source 1 Channel Offset", parameter::Type::Long});
          params.back().setInstantiation(true).setMin(0).setDefault(0);
          
          params.emplace_back(desc_type{"numchans1","Source 1 Channels", parameter::Type::Long});
          params.back().setInstantiation(true).setMin(-1).setDefault(-1);
          
          params.emplace_back(desc_type{"src1gain","Source 1 Gain", parameter::Type::Float});
          params.back().setInstantiation(false).setDefault(1);
          
          params.emplace_back(desc_type{"src2","Second Source Buffer", parameter::Type::Buffer});
          params.back().setInstantiation(true);
          
          params.emplace_back(desc_type{"offsetframes2","Source 2 Offset", parameter::Type::Long});
          params.back().setInstantiation(true).setMin(0).setDefault(0);
          
          params.emplace_back(desc_type{"numframes2","Source 2 Frames", parameter::Type::Long});
          params.back().setInstantiation(true).setMin(-1).setDefault(-1);
          
          params.emplace_back(desc_type{"offsetchans2","Source 2 Channel Offset", parameter::Type::Long});
          params.back().setInstantiation(true).setMin(0).setDefault(0);
          
          params.emplace_back(desc_type{"numchans2","Source 2 Channels", parameter::Type::Long});
          params.back().setInstantiation(true).setMin(-1).setDefault(-1);
          
          params.emplace_back(desc_type{"src2gain","Source 2 Gain", parameter::Type::Float});
          params.back().setInstantiation(false).setDefault(1);
          
          params.emplace_back(desc_type{"dstbuf","Destination Buffer", parameter::Type::Buffer});
          params.back().setInstantiation(false);
        }
        return params;
      }
      
      void newParameterSet()
      {
        mParams.clear();
        mParams.reserve(getParamDescriptors().size());
        //Note: I'm pretty sure I want auto's copy behaviour here
        for(auto p: getParamDescriptors())
          mParams.emplace_back( parameter::Instance(p));
      }
      
      /**
       Go over the supplied parameter values and ensure that they are sensible
       Args
       source buffer [offset numframes offsetchan numchan src gain]
      **/
      std::tuple<bool,std::string,ProcessModel> sanityCheck()
      {
        ProcessModel model;
        const std::vector<parameter::Descriptor>& desc = getParamDescriptors();
        //First, let's make sure that we have a complete of parameters of the right sort
        bool sensible = std::equal(mParams.begin(), mParams.end(),desc.begin(),
          [](const param_type& i, const parameter::Descriptor& d)
          {
            return i.getDescriptor() == d;
          });
        
        if(! sensible || (desc.size() != mParams.size()))
        {
          return {false, "Invalid params passed. Were these generated with newParameterSet()?", model };
        }
        
        size_t bufCount = 0;
        std::unordered_set<parameter::BufferAdaptor*> uniqueBuffers;
        //First round of buffer checks
        //Source buffer is mandatory, and should exist
        parameter::BufferAdaptor::Access src1(mParams[0].getBuffer());
        parameter::BufferAdaptor::Access src2(mParams[6].getBuffer());

        if(!(src1.valid() && src2.valid()))
        {
          return  { false, "At least one source buffer doesn't exist or can't be accessed.", model };
        }
        
        if(src1.numFrames() == 0 || src2.numFrames() == 0)
        {
          return {false, "At least one source buffer is empty",model};
        }
     
        for(auto&& p: mParams)
        {
          switch(p.getDescriptor().getType())
          {
            case parameter::Type::Buffer:
              //If we've been handed a buffer that we're expecting, then it should exist
              if(p.hasChanged() && p.getBuffer())
              {
                parameter::BufferAdaptor::Access b(p.getBuffer());
                if(!b.valid())
                 {
                   std::ostringstream ss;
                   ss << "Buffer given for " << p.getDescriptor().getName() << " doesn't exist.";
                
                   return {false, ss.str(), model};
                 }
                ++bufCount;
                uniqueBuffers.insert(p.getBuffer());
              }
            default:
              continue;
          }
        }
 
        if(bufCount < 3)
        {
          return { false, "Expecting three valid buffers", model};
        }
//
//        if(bufCount > uniqueBuffers.size())
//        {
//          return {false, "One or more buffers are the same. They all need to be distinct", model};
//        }
        
        
        //Now scan everything for range, until we hit a problem
        //TODO Factor into parameter::instance
        for(auto&& p: mParams)
        {
          parameter::Descriptor d = p.getDescriptor();
          bool rangeOk;
          parameter::Instance::RangeErrorType errorType;
          std::tie(rangeOk, errorType) = p.checkRange();
          if (!rangeOk)
          {
            std::ostringstream msg;
            msg << "Parameter " << d.getName();
            switch (errorType)
            {
              case parameter::Instance::RangeErrorType::Min:
                msg << " value below minimum (" << d.getMin() << ")";
                break;
              case parameter::Instance::RangeErrorType::Max:
                msg << " value above maximum (" << d.getMin() << ")";
              default:
                assert(false && "This should be unreachable");
            }
            return { false, msg.str(), model};
          }

        }
        
  
        //Check the size of our buffers
//        parameter::BufferAdaptor* src= params[0].getBuffer();
        
        long srcOffset     = parameter::lookupParam("offsetframes1",mParams).getLong();
        long srcFrames     = parameter::lookupParam("numframes1",   mParams).getLong();
        long srcChanOffset = parameter::lookupParam("offsetchans1", mParams).getLong();
        long srcChans      = parameter::lookupParam("numchans1",    mParams).getLong();
        
        
        //Ensure that the source buffer can deliver
        if(srcFrames > 0 ? (src1.numFrames() < (srcOffset + srcFrames)) : (src1.numFrames() < srcOffset))
        {
          return  { false, "Source buffer 1 not long enough for given offset and frame count",model};
        }
        
        if((srcChans > 0) ? (src1.numChans() < (srcChanOffset + srcChans)) : (src1.numChans() < srcChanOffset))
        {
          return {false, "Source buffer 1 doesn't have enough channels for given offset and channel count", model};
        }
        
        //At this point, we're happy with the source buffer
        model.src1           = mParams[0].getBuffer();
        model.offset[0]        = srcOffset;
        model.frames[0]        = srcFrames > 0 ? srcFrames : src1.numFrames() - model.offset[0];
        model.channelOffset[0] = srcChanOffset;
        model.channels[0]      = srcChans >  0 ? srcChans  : src1.numChans() - model.channelOffset[0];
        model.gain[0]    = parameter::lookupParam("src1gain",mParams).getFloat();
        
        srcOffset     = parameter::lookupParam("offsetframes2",mParams).getLong();
        srcFrames     = parameter::lookupParam("numframes2",   mParams).getLong();
        srcChanOffset = parameter::lookupParam("offsetchans2", mParams).getLong();
        srcChans      = parameter::lookupParam("numchans2",    mParams).getLong();
        
        //Ensure that the source buffer can deliver
        if(srcFrames > 0 ? (src2.numFrames() < (srcOffset + srcFrames)) : (src2.numFrames() < srcOffset))
        {
          return  { false, "Source buffer 2 not long enough for given offset and frame count",model};
        }
        
        if((srcChans > 0) ? (src2.numChans() < (srcChanOffset + srcChans)) : (src2.numChans() < srcChanOffset))
        {
          return {false, "Source buffer 2 doesn't have enough channels for given offset and channel count", model};
        }
        
        //At this point, we're happy with the source buffer
        model.src2             = mParams[6].getBuffer();
        model.offset[1]        = srcOffset;
        model.frames[1]        = srcFrames > 0 ? srcFrames : src2.numFrames() - model.offset[1];
        model.channelOffset[1] = srcChanOffset;
        model.channels[1]      = srcChans >  0 ? srcChans  : src2.numChans() - model.channelOffset[1];
        model.gain[1]    = parameter::lookupParam("src2gain",mParams).getFloat();
        
        parameter::BufferAdaptor::Access dst(mParams[12].getBuffer());
        if(! dst.valid())
        {
          return {false,"Destination buffer invalid",model};
        }
        
        model.dst = mParams[12].getBuffer();
        
        //We made it
        return {true, "Everything is lovely",model};
      }
      
      
      //No, you may not  copy this, or move this
      BufferComposeClient(BufferComposeClient&)=delete;
      BufferComposeClient(BufferComposeClient&&)=delete;
      BufferComposeClient operator=(BufferComposeClient&)=delete;
      BufferComposeClient operator=(BufferComposeClient&&)=delete;
      
      BufferComposeClient(){
        newParameterSet(); 
      }
   
      void process(ProcessModel model)
      {
        parameter::BufferAdaptor::Access src1(model.src1);
        parameter::BufferAdaptor::Access src2(model.src2);
        parameter::BufferAdaptor::Access dst(model.dst);
        
        size_t chans = *std::max_element(model.channels, model.channels + 2);
        size_t frames = *std::min_element(model.frames,model.frames + 2);
        dst.resize(frames,chans,1);
   
        for(size_t i = 0; i < chans; ++i)
        {
          FluidTensor<double,1> dstChan(frames);
          dstChan = src1.samps(model.offset[0], frames, model.channelOffset[0] + (i % src1.numChans()),1).col(0);
          double g1 = model.gain[0];
          dstChan.apply([g1](double& x){
            x *= g1;
          });
          
          FluidTensor<double,1> src2Frames(frames);
          src2Frames = src2.samps(model.offset[1], frames, model.channelOffset[1] + (i % src2.numChans()),1).col(0);
          double g2 = model.gain[1];
          src2Frames.apply([g2](double& x){
            x *= g2;
          });
          
          dstChan.apply(src2Frames,[](double& x, double& y){
            x += y;
          });
          
          dst.samps(i) = dstChan;
        }
        
      }
      std::vector<parameter::Instance>& getParams()
      {
        return mParams;
      }
      
    private:
      std::vector<parameter::Instance> mParams;
    };
  } //namespace buf
} //namesapce fluid
