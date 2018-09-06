#pragma once


#include "clients/common/FluidParams.hpp"
#include "clients/common/STFTCheckParams.hpp"
#include "data/FluidTensor.hpp"
#include "data/FluidBuffers.hpp"
#include "algorithms/NMF.hpp"
#include "algorithms/STFT.hpp"
#include "algorithms/RatioMask.hpp"


#include <algorithm> //for max_element
#include <string>
#include <sstream> //for ostringstream
#include <utility> //for std make_pair
#include <unordered_set>
#include <vector> //for containers of params, and for checking things


using fluid::FluidTensor;
using fluid::nmf::NMF;
using fluid::stft::STFT;
using fluid::stft::ISTFT;
using fluid::stft::Spectrogram;

namespace fluid {
  namespace nmf{
    
    /**
     Integration class for doing NMF filtering and resynthesis
     **/
    class NMFClient
    {
      using desc_type = parameter::Descriptor;
      using param_type = parameter::Instance;
    public:
      struct ProcessModel
      {
        bool fixDictionaries;
        bool fixActivations;
        bool seedDictionaries;
        bool seedActivations;
        bool returnDictionaries;
        bool returnActivations;
        
        size_t windowSize;
        size_t hopSize;
        size_t fftSize;

        size_t rank;
        size_t iterations;
        bool resynthesise;
        
        size_t frames;
        size_t offset;
        
        size_t channels;
        size_t channelOffset;
        
        parameter::BufferAdaptor* src = 0 ;
        parameter::BufferAdaptor* resynth = 0 ;
        parameter::BufferAdaptor* dict  = 0 ;
        parameter::BufferAdaptor* act = 0;
      };
      
      static const std::vector<parameter::Descriptor>& getParamDescriptors()
      {
        static std::vector<desc_type> params;
        if(params.empty())
        {
          params.emplace_back(desc_type{"src","Source Buffer", parameter::Type::Buffer});
          params.back().setInstantiation(true);
          
          params.emplace_back(desc_type{"offsetframes","Source Offset", parameter::Type::Long});
          params.back().setInstantiation(true).setMin(0).setDefault(0);

          params.emplace_back(desc_type{"numframes","Source Frames", parameter::Type::Long});
          params.back().setInstantiation(true).setMin(-1).setDefault(-1);
          
          params.emplace_back(desc_type{"offsetchans","Source Channel Offset", parameter::Type::Long});
          params.back().setInstantiation(true).setMin(0).setDefault(0);
          
          params.emplace_back(desc_type{"numchans","Source Channels", parameter::Type::Long});
          params.back().setInstantiation(true).setMin(-1).setDefault(-1);

          params.emplace_back(desc_type{"resynthbuf","Resynthesis Buffer", parameter::Type::Buffer});
          params.back().setInstantiation(false);

          params.emplace_back(desc_type{"filterbuf","Filters Buffer", parameter::Type::Buffer});
          params.back().setInstantiation(false);

          params.emplace_back(desc_type{"filterupdate","Filter Update",  parameter::Type::Long});
          params.back().setInstantiation(false).setMin(0).setMax(2).setDefault(0);
          
          params.emplace_back(desc_type{"envbuf","Envelopes Buffer", parameter::Type::Buffer});
          params.back().setInstantiation(false);
        
          params.emplace_back(desc_type{"envupdate","Activation Update",  parameter::Type::Long});
          params.back().setInstantiation(false).setMin(0).setMax(2).setDefault(0);
          
          params.emplace_back(desc_type{"rank","Rank", parameter::Type::Long});
          params.back().setMin(1).setDefault(1);
          
          params.emplace_back(desc_type{"iterations","Iterations", parameter::Type::Long});
          params.back().setInstantiation(false).setMin(1).setDefault(100);
          
          params.emplace_back(desc_type{"winsize","Window Size", parameter::Type::Long});
          params.back().setMin(4).setDefault(1024);

          params.emplace_back(desc_type{"hopsize","Hop Size", parameter::Type::Long});
          params.back().setMin(1).setDefault(256);

          params.emplace_back(desc_type{"fftsize","FFT Size", parameter::Type::Long});
          params.back().setMin(-1).setDefault(-1);
        }
        return params;
      }
      
      
      /**
       Go over the supplied parameter values and ensure that they are sensible
       No hygiene checking of buffers is done here (like whether they exist). It needs to be done in the host code, until I've worked out something cleverer
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
        
        parameter::BufferAdaptor::Access src(mParams[0].getBuffer());
          
        if(!src.valid())
        {
          return  { false, "Source buffer doesn't exist or can't be accessed.", model };
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
 
        if(bufCount < 2)
        {
          return { false, "Expecting at least two valid buffers", model};
        }
        
        if(bufCount > uniqueBuffers.size())
        {
          return {false, "One or more buffers are the same. They all need to be distinct", model};
        }
        
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
                msg << " value below minimum(" << d.getMin() << ")";
                break;
              case parameter::Instance::RangeErrorType::Max:
                msg << " value above maximum(" << d.getMin() << ")";
              default:
                assert(false && "This should be unreachable");
            }
            return { false, msg.str(), model};
          }
        }
        
        //Now make sense of the overwrite parameter
        //if 0 (normal) then we expect to reallocate both dicts and acts and init randomly
        //if 1 (seeding) then the buffer should be present, allocated, and will mutate
        //if 2 (matching) then the buffer should be present, allocated and won't mutate
        //having both == 2 makes no sense
        
        parameter::Instance& dictUpdateRule = parameter::lookupParam("filterupdate", mParams);
        parameter::Instance& actUpdateRule  = parameter::lookupParam("envupdate", mParams);
        
        if(dictUpdateRule.getLong() == 2 && actUpdateRule.getLong() == 2)
        {
          return {false, "It makes no sense to update neither the dictionaries nor the activaitons", model};
        }
        
        model.seedDictionaries = (dictUpdateRule.getLong() > 0);
        model.fixDictionaries  = (dictUpdateRule.getLong() == 2);
        model.seedActivations  = (actUpdateRule.getLong() > 0);
        model.fixActivations   = (actUpdateRule.getLong() == 2);
        
        //Check the size of our buffers
        
        long srcOffset     = parameter::lookupParam("offsetframes",         mParams).getLong();
        long srcFrames     = parameter::lookupParam("numframes",         mParams).getLong();
        long srcChanOffset = parameter::lookupParam("offsetchans", mParams).getLong();
        long srcChans      = parameter::lookupParam("numchans",       mParams).getLong();
        
        //Ensure that the source buffer can deliver
        if(srcFrames > 0 ? (src.numFrames() < (srcOffset + srcFrames)) : (src.numFrames() < srcOffset))
        {
          return  { false, "Source buffer not long enough for given offset and frame count",model};
        }
        
        if((srcChans > 0) ? (src.numChans() < (srcChanOffset + srcChans)) : (src.numChans() < srcChanOffset))
        {
          return {false, "Source buffer doesn't have enough channels for given offset and channel count", model};
        }
        
        //At this point, we're happy with the source buffer
        model.src           = mParams[0].getBuffer();
        model.offset        = srcChanOffset;
        model.frames        = srcFrames > 0 ? srcFrames : src.numFrames() - model.offset;
        model.channelOffset = srcChanOffset;
        model.channels      = srcChans >  0 ? srcChans  : src.numChans() - model.channelOffset;
    
        //Check the FFT args
        
        parameter::Instance& windowSize = lookupParam("winsize", mParams);
        parameter::Instance& fftSize    = lookupParam("fftsize", mParams);
        parameter::Instance& hopSize    = lookupParam("hopsize", mParams);
      
        auto fftOk = parameter::checkFFTArguments(windowSize,hopSize,fftSize);
        
        if(!std::get<0>(fftOk))
        {
          //Glue the fft error message together with our model
          return std::tuple_cat(fftOk,std::tie(model));
        }
      
        model.fftSize    = fftSize.getLong();
        model.windowSize = windowSize.getLong();
        model.hopSize    = hopSize.getLong();
        model.rank       = parameter::lookupParam("rank",mParams).getLong();
        model.iterations = parameter::lookupParam("iterations",mParams).getLong();
        
        parameter::Instance& resynth = parameter::lookupParam("resynthbuf", mParams);
        parameter::BufferAdaptor::Access resynthBuf(resynth.getBuffer());
        
        if(resynth.hasChanged() && (!resynth.getBuffer() || !resynthBuf.valid()))
        {
          return {false, "Invalid resynthesis buffer supplied", model};
        }
        
        model.resynthesise = resynth.hasChanged() && resynthBuf.valid();
        if(model.resynthesise)
        {
          resynthBuf.resize(src.numFrames(), src.numChans(), model.rank);
          model.resynth = resynth.getBuffer();
        }
        
        parameter::Instance& dict = parameter::lookupParam("filterbuf", mParams);
        parameter::BufferAdaptor::Access dictBuf(dict.getBuffer());

        if(dict.hasChanged() && (!dict.getBuffer() || !dictBuf.valid()))
        {
          return {false, "Invalid filters buffer supplied",model};
        }
        if(model.fixDictionaries || model.seedDictionaries)
        {
          if(!dict.hasChanged())
            return {false,"No dictionary buffer given, but one needed for seeding or matching",model};          
          //Prepared Dictionary buffer needs to be (fftSize/2 + 1) by (rank * srcChans)
          if(dictBuf.numFrames() != (model.fftSize / 2) + 1 || dictBuf.numChans() != model.rank * model.channels)
            return {false, "Pre-prepared dictionary buffer must be [(FFTSize / 2) + 1] frames long, and have [rank] * [channels] channels",model };
        } else {
          if(dict.hasChanged()) //a valid buffer has been designated and needs to be resized
          {
            dictBuf.resize(model.fftSize/2 + 1, model.channels, model.rank);
          }
        }
        model.returnDictionaries = dict.hasChanged();
        model.dict = dict.getBuffer(); 
        
        parameter::Instance& act = parameter::lookupParam("envbuf", mParams);
        parameter::BufferAdaptor::Access actBuf(act.getBuffer());

        if(act.hasChanged() && (!act.getBuffer() && !actBuf.valid()))
        {
          return {false, "Invalid envelope buffer supplied",model};
        }
        if(model.fixActivations || model.seedActivations)
        {
          if(!act.hasChanged())
            return {false, "No dictionary buffer given, but one needed for seeding or matching", model};
          
          //Prepared activation buffer needs to be (src Frames / hop size + 1) by (rank * srcChans)
          if(actBuf.numFrames() != (model.fftSize / 2) + 1 || actBuf.numChans() != model.rank * src.numChans())
          {
            return {false,"Pre-prepared activation buffer must be [(num samples / hop size) + 1] frames long, and have [rank] * [channels] channels", model};
          }
        } else {
          if(act.hasChanged())
          {
            actBuf.resize((model.frames / model.hopSize) + 1, model.channels, model.rank);
          }
        }
        model.returnActivations = act.hasChanged();
        model.act = act.getBuffer();

        //We made it
        return {true, "Everything is lovely",model};
      }
      
      
      NMFClient(){
        newParameterSet();
      };
      // no copy this, nor move this

      NMFClient(NMFClient&)=delete;
      NMFClient(NMFClient&&)=delete;
      NMFClient operator=(NMFClient&)=delete;
      NMFClient operator=(NMFClient&&)=delete;
      
      /**
       You may constrct one by supplying some senisble numbers here
       rank: NMF rank
       iterations: max nmf iterations
       fft_size: power 2 pls
//       **/
//        NMFClient(ProcessModel model):
//        mArguments(model)
     
      
      ~NMFClient()= default;
      

      
//      /**
//       Call this before pushing / processing / pulling
//       to prepare buffers to correct size
//       **/
//      void setSourceSize(size_t source_frames)
//      {
//        mSource.set_host_buffer_size(source_frames);
//        mSinkResynth.set_host_buffer_size(source_frames);
//      }
      
      /***
       Take some data, NMF it
       ***/
      void process(ProcessModel model)
      {
        
        mArguments = model;
        
        parameter::BufferAdaptor::Access src(model.src);
        parameter::BufferAdaptor::Access dict(model.dict);
        parameter::BufferAdaptor::Access act(model.act);
        parameter::BufferAdaptor::Access resynth(model.resynth);

        
//        mSource.set_host_buffer_size(mArguments.frames);
//        mSource.reset();
//        mSource.push(*mArguments.src);
        
//        mAudioBuffers.resize(mRank,data.extent(0));
        
//        mHasProcessed = false;
//        mHasResynthed = false;
        stft::STFT stft(mArguments.windowSize,mArguments.fftSize,mArguments.hopSize);
        //Copy input buffer
        RealMatrix sourceData(src.samps(mArguments.offset, mArguments.frames, mArguments.channelOffset, mArguments.channels));
        //TODO: get rid of need for this
        //Either: do the whole process loop here via process_frame
        //Or: change sig of stft.process() to take a view
        RealVector tmp(mArguments.frames);
        for(size_t i = 0; i < mArguments.channels; ++i)
        {
          tmp = sourceData.col(i);
          stft::Spectrogram spec = stft.process(tmp);
          
          //TODO: Add seeding with pre-formed W & H to NMF
          nmf::NMF nmf(mArguments.rank,mArguments.iterations);
          nmf::NMFModel m = nmf.process(spec.getMagnitude());
          
          //Write W?
          if(mArguments.returnDictionaries)
          {
            auto dictionaries = m.getW();
            for (size_t j = 0; j < mArguments.rank; ++j)
              dict.samps(i,j) = dictionaries.col(j);
              //mArguments.dict->col(j + (i * mArguments.rank)) = dictionaries.col(j);
          }
          
          //Write H? Need to normalise also
          if(mArguments.returnActivations)
          {
            auto activations = m.getH();
            
            double maxH = *std::max_element(activations.begin(), activations.end());
            
            double scale = 1. / (maxH);
            
            for (size_t j = 0; j < mArguments.rank; ++j)
            {
              auto acts = act.samps(i,j);
              acts = activations.row(j);
              acts.apply([scale](float& x){
                x *= scale;
              });
            }
          }

          if(mArguments.resynthesise)
          {
            ratiomask::RatioMask mask(m.getMixEstimate(),1);
            stft::ISTFT  istft(mArguments.windowSize, mArguments.fftSize, mArguments.hopSize);
            for (size_t j = 0; j < mArguments.rank; ++j)
            {
              auto estimate = m.getEstimate(j);
              stft::Spectrogram result(mask.process(spec.mData, estimate));
              auto audio = istft.process(result);
              resynth.samps(i,j) = audio(fluid::slice(0,mArguments.frames));
            }
          }
        }
      }
      
      std::vector<parameter::Instance>& getParams()
      {
        return mParams;
      }
      
      
    private:
   
      ProcessModel mArguments;
      fluid::nmf::NMFModel mModel;
      FluidTensor<double,2> mAudioBuffers;
    
      void newParameterSet()
      {
        mParams.clear();
        //Note: I'm pretty sure I want auto's copy behaviour here
        for(auto p: getParamDescriptors())
          mParams.emplace_back(parameter::Instance(p));
      }
      
      std::vector<parameter::Instance> mParams;
      
      
    };
  } //namespace max
} //namesapce fluid
