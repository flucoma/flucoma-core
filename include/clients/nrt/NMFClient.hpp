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
        
        parameter::BufferAdaptor* src;
        parameter::BufferAdaptor* resynth;
        parameter::BufferAdaptor* dict;
        parameter::BufferAdaptor* act;
      };
      
      static const std::vector<parameter::Descriptor>& getParamDescriptors()
      {
        static std::vector<desc_type> params;
        if(params.empty())
        {
          params.emplace_back(desc_type{"Source Buffer", parameter::Type::Buffer});
          params.back().setInstantiation(true);
          
          params.emplace_back(desc_type{"Source Offset", parameter::Type::Long});
          params.back().setInstantiation(true).setMin(0).setDefault(0);

          params.emplace_back(desc_type{"Source Frames", parameter::Type::Long});
          params.back().setInstantiation(true).setMin(-1).setDefault(-1);
          
          params.emplace_back(desc_type{"Source Channel Offset", parameter::Type::Long});
          params.back().setInstantiation(true).setMin(0).setDefault(0);
          
          params.emplace_back(desc_type{"Source Channels", parameter::Type::Long});
          params.back().setInstantiation(true).setMin(-1).setDefault(-1);

          params.emplace_back(desc_type{"Resynthesis Buffer", parameter::Type::Buffer});
          params.back().setInstantiation(true);

          params.emplace_back(desc_type{"Dictionary Buffer", parameter::Type::Buffer});
          params.back().setInstantiation(true);

          params.emplace_back(desc_type{"Dictionary Update",  parameter::Type::Long});
          params.back().setInstantiation(true).setMin(0).setMax(2).setDefault(0);
          
          params.emplace_back(desc_type{"Activation Buffer", parameter::Type::Buffer});
          params.back().setInstantiation(true);
        
          params.emplace_back(desc_type{"Activation Update",  parameter::Type::Long});
          params.back().setInstantiation(true).setMin(0).setMax(2).setDefault(0);
          
          params.emplace_back(desc_type{"Rank",  parameter::Type::Long});
          params.back().setMin(1).setDefault(1);
          
          params.emplace_back(desc_type{"Iterations", parameter::Type::Long});
          params.back().setMin(1).setDefault(100);
          
          params.emplace_back(desc_type{"Window Size", parameter::Type::Long});
          params.back().setMin(4).setDefault(1024);

          params.emplace_back(desc_type{"Hop Size", parameter::Type::Long});
          params.back().setMin(1).setDefault(256);

          params.emplace_back(desc_type{"FFT Size", parameter::Type::Long});
          params.back().setMin(4).setDefault(1024);
        }
        return params;
      }
      
      static const std::vector<param_type> newParameterSet()
      {
        std::vector<param_type> newParams;
        //Note: I'm pretty sure I want auto's copy behaviour here
        for(auto p: getParamDescriptors())
          newParams.emplace_back( parameter::Instance(p));
        return newParams;
      }
      
      /**
       Go over the supplied parameter values and ensure that they are sensible
       No hygiene checking of buffers is done here (like whether they exist). It needs to be done in the host code, until I've worked out something cleverer
      **/
      static std::tuple<bool,std::string,ProcessModel> sanityCheck(std::vector<param_type>& params)
      {
        ProcessModel model;
        const std::vector<parameter::Descriptor>& desc = getParamDescriptors();
        //First, let's make sure that we have a complete of parameters of the right sort
        bool sensible = std::equal(params.begin(), params.end(),desc.begin(),
          [](const param_type& i, const parameter::Descriptor& d)
          {
            return i.getDescriptor() == d;
          });
        
        if(! sensible || (desc.size() != params.size()))
        {
          return {false, "Invalid params passed. Were these generated with newParameterSet()?", model };
        }
        
        size_t bufCount = 0;
        std::unordered_set<parameter::BufferAdaptor*> uniqueBuffers;
        //First round of buffer checks
        //Source buffer is mandatory, and should exist
        if(!params[0].getBuffer() || !params[0].getBuffer()->valid())
        {
          return  { false, "Source buffer doesn't exist or can't be accessed.", model };
        }
        
      
        for(auto&& p: params)
        {
          switch(p.getDescriptor().getType())
          {
            case parameter::Type::Buffer:
              //If we've been handed a buffer that we're expecting, then it should exist
              if(p.hasChanged() && p.getBuffer())
              {
                if(!p.getBuffer()->valid())
             
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
        for(auto&& p: params)
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
        
        parameter::Instance dictUpdateRule = parameter::lookupParam("Dictionary Update", params);
        parameter::Instance actUpdateRule  = parameter::lookupParam("Activation Update", params);
        
        if(dictUpdateRule.getLong() == 2 && actUpdateRule.getLong() == 2)
        {
          return {false, "It makes no sense to update neither the dictionaries nor the activaitons", model};
        }
        
        model.seedDictionaries = (dictUpdateRule.getLong() > 0);
        model.fixDictionaries  = (dictUpdateRule.getLong() == 2);
        model.seedActivations  = (actUpdateRule.getLong() > 0);
        model.fixActivations   = (actUpdateRule.getLong() == 2);
        
        
        //Check the size of our buffers
        parameter::BufferAdaptor* src= params[0].getBuffer();
        
        long srcOffset     = parameter::lookupParam("Source Offset",         params).getLong();
        long srcFrames     = parameter::lookupParam("Source Frames",         params).getLong();
        long srcChanOffset = parameter::lookupParam("Source Channel Offset", params).getLong();
        long srcChans      = parameter::lookupParam("Source Channels",       params).getLong();
        
        //Ensure that the source buffer can deliver
        if(srcFrames > 0 ? (src->numSamps() < (srcOffset + srcFrames)) : (src->numSamps() < srcOffset))
        {
          return  { false, "Source buffer not long enough for given offset and frame count",model};
        }
        
        if((srcChans > 0) ? (src->numChans() < (srcChanOffset + srcChans)) : (src->numChans() < srcChanOffset))
        {
          return {false, "Source buffer doesn't have enough channels for given offset and channel count", model};
        }
        
        //At this point, we're happy with the source buffer
        model.src           = src;
        model.offset        = srcChanOffset;
        model.frames        = srcFrames > 0 ? srcFrames : src->numSamps() - model.offset;
        model.channelOffset = srcChanOffset;
        model.channels      = srcChans >  0 ? srcChans  : src->numChans() - model.channelOffset;
        
        
        //Check the FFT args
        
        parameter::Instance windowSize = lookupParam("Window Size", params);
        parameter::Instance fftSize    = lookupParam("FFT Size", params);
        parameter::Instance hopSize    = lookupParam("Hop Size", params);
      
        auto fftOk = parameter::checkFFTArguments(windowSize,hopSize,fftSize);
        
        if(!std::get<0>(fftOk))
        {
          //Glue the fft error message together with our model
          return std::tuple_cat(fftOk,std::tie(model));
        }
      
        model.fftSize    = fftSize.getLong();
        model.windowSize = windowSize.getLong();
        model.hopSize    = hopSize.getLong();
        model.rank       = parameter::lookupParam("Rank",params).getLong();
        model.iterations = parameter::lookupParam("Iterations",params).getLong();
        
        parameter::Instance resynth = parameter::lookupParam("Resynthesis Buffer", params);
        
        if(resynth.hasChanged() && (!resynth.getBuffer() ||!resynth.getBuffer()->valid()))
        {
          return {false, "Invalid resynthesis buffer supplied", model};
        }
        
        model.resynthesise  = resynth.hasChanged() && resynth.getBuffer() && resynth.getBuffer()->valid();
        if(model.resynthesise)
        {
          resynth.getBuffer()->resize( src->numSamps(), src->numChans(), model.rank);
          model.resynth = resynth.getBuffer();
        }
        
        parameter::Instance dict = parameter::lookupParam("Dictionary Buffer", params);
        if(dict.hasChanged() && (!dict.getBuffer() || !dict.getBuffer()->valid()))
        {
          return {false, "Invalid filters buffer supplied",model};
        }
        if(model.fixDictionaries || model.seedDictionaries)
        {
          if(!dict.hasChanged())
            return {false,"No dictionary buffer given, but one needed for seeding or matching",model};          
          //Prepared Dictionary buffer needs to be (fftSize/2 + 1) by (rank * srcChans)
          if(dict.getBuffer()->numSamps() != (model.fftSize / 2) + 1 || dict.getBuffer()->numChans() != model.rank * model.channels)
            return {false, "Pre-prepared dictionary buffer must be [(FFTSize / 2) + 1] frames long, and have [rank] * [channels] channels",model };
        } else {
          if(dict.hasChanged()) //a valid buffer has been designated and needs to be resized
          {
            dict.getBuffer()->resize(model.fftSize/2 + 1, model.channels, model.rank);
          }
        }
        model.returnDictionaries = dict.hasChanged();
        model.dict = dict.getBuffer(); 
        
        parameter::Instance act = parameter::lookupParam("Activation Buffer", params);
        if(act.hasChanged() && (!act.getBuffer() && !act.getBuffer()->valid()))
        {
          return {false, "Invalid envelope buffer supplied",model};
        }
        if(model.fixActivations || model.seedActivations)
        {
          if(!act.hasChanged())
            return {false, "No dictionary buffer given, but one needed for seeding or matching", model};
          
          //Prepared activation buffer needs to be (src Frames / hop size + 1) by (rank * srcChans)
          if(act.getBuffer()->numSamps() != (model.fftSize / 2) + 1 || act.getBuffer()->numChans() != model.rank * src->numChans())
          {
            return {false,"Pre-prepared activation buffer must be [(num samples / hop size) + 1] frames long, and have [rank] * [channels] channels", model};
          }
        } else {
          if(act.hasChanged())
          {
            act.getBuffer()->resize((model.frames / model.hopSize) + 1, model.channels, model.rank);
          }
        }
        model.returnActivations = act.hasChanged();
        model.act = act.getBuffer();

        //We made it
        return {true, "Everything is lovely",model};
      }
      
      
      //No, you may not construct an empty instance, or copy this, or move this
      NMFClient() = delete;
      NMFClient(NMFClient&)=delete;
      NMFClient(NMFClient&&)=delete;
      NMFClient operator=(NMFClient&)=delete;
      NMFClient operator=(NMFClient&&)=delete;
      
      /**
       You may constrct one by supplying some senisble numbers here
       rank: NMF rank
       iterations: max nmf iterations
       fft_size: power 2 pls
       **/
        NMFClient(ProcessModel model):
      mSource(model.channels), mSinkResynth(model.rank * model.channels), mSinkDictionaries(model.rank * model.channels),
      mSinkActivations(model.rank * model.channels), mArguments(model)      
      {}
      
      ~NMFClient()= default;
      //Not implemented
      //void reset();
      //bool isReady() const;
      
      /**
       Call this before pushing / processing / pulling
       to prepare buffers to correct size
       **/
      void setSourceSize(size_t source_frames)
      {
        mSource.set_host_buffer_size(source_frames);
        mSinkResynth.set_host_buffer_size(source_frames);
        //mSinkDictionaries.set_host_buffer_size(<#size_t n#>)
      }
      
      
      /***
       Take some data, NMF it
       ***/
      void process()
      {
//        mSource.set_host_buffer_size(mArguments.frames);
//        mSource.reset();
//        mSource.push(*mArguments.src);
        
//        mAudioBuffers.resize(mRank,data.extent(0));
        
//        mHasProcessed = false;
//        mHasResynthed = false;
        stft::STFT stft(mArguments.windowSize,mArguments.fftSize,mArguments.hopSize);
        //Copy input buffer
        mArguments.src->acquire();
        RealMatrix sourceData(*mArguments.src);
        mArguments.src->release();
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
            mArguments.dict->acquire();
            auto dictionaries = m.getW();
            for (size_t j = 0; j < mArguments.rank; ++j)
              mArguments.dict->col(j + (i * mArguments.rank)) = dictionaries.col(j);
            mArguments.dict->release();
          }
          
          //Write H? Need to normalise also
          if(mArguments.returnActivations)
          {
            mArguments.act->acquire();
            auto activations = m.getH();
            
            double maxH = *std::max_element(activations.begin(), activations.end());
            
            double scale = maxH / mArguments.fftSize;
            
            for (size_t j = 0; j < mArguments.rank; ++j)
            {
              mArguments.act->col(j + (i * mArguments.rank)) = activations.row(j);
              mArguments.act->col(j + (i * mArguments.rank)).apply([scale](float& x){
                x *= scale;
              });
            }
            mArguments.act->release();
          }

          if(mArguments.resynthesise)
          {
            mArguments.resynth->acquire();
            ratiomask::RatioMask mask(m.getMixEstimate(),1);
            stft::ISTFT  istft(mArguments.windowSize, mArguments.fftSize, mArguments.hopSize);
            for (size_t j = 0; j < mArguments.rank; ++j)
            {
              auto estimate = m.getEstimate(j);
              stft::Spectrogram result(mask.process(spec.mData, estimate));
              auto audio = istft.process(result);
              mArguments.resynth->col(j + (i * mArguments.rank)) = audio(fluid::slice(0,mArguments.frames));
            }
            mArguments.resynth->release();
          }
          
          
        }
        
        
//        stft::Spectrogram spec = stft.process(mArguments.src->row(0));
//        RealMatrix mag = spec.getMagnitude();
//        nmf::NMF nmf(mArguments.rank,mArguments.iterations);
//        nmf::NMFModel m = nmf.process(spec.getMagnitude());
////        mHasProcessed = true;
//
//        if(mArguments.resynthesise)
//        {
//          ratiomask::RatioMask mask(mModel.getMixEstimate(),1);
//          stft::ISTFT istft(mWindowSize, mFFTSize, mHopSize);
//          for(int i = 0; i < mRank; ++i)
//          {
//            RealMatrix estimate = mModel.getEstimate(i);
//            stft::Spectrogram result(mask.process(spec.mData, estimate));
//            RealVector audio = istft.process(result);
//              mAudioBuffers.row(i) = audio(fluid::slice(0,data.extent(0)));
//          }
//          mHasResynthed = true;
//        }
      }
      
//      /***
//       Report the size of a dictionary, in bins (= fft_size/2)
//       ***/
//      size_t dictionary_size() const
//      {
//        return mHasProcessed ? mModel.getW().extent(0) : 0 ;
//      }
//
//      /***
//       Report the length of an activation, in frames
//       ***/
//      size_t activations_length() const{
//        return mHasProcessed ? mModel.getH().extent(1) : 0;
//      }
//
//      /***
//       Report the number of sources (i.e. the rank
//       ***/
//      size_t num_sources() const
//      {
//        return mHasResynthed ? mAudioBuffers.size() : 0;
//      }
//      //        size_t rank() const;
//
//      /***
//       Retreive the dictionary at the given index
//       ***/
//      const FluidTensorView<double, 1> dictionary(const size_t idx) const
//      {
//        assert(mHasProcessed && idx < mModel.W.cols());
//        return mModel.getW().col(idx);
//      }
//
//      /***
//       Retreive the activation at the given index
//       ***/
//      const FluidTensorView<double, 1> activation(const size_t idx) const
//      {
//        assert(mHasProcessed && idx < mModel.H.rows());
//        return mModel.getH().row(idx);
//      }
//
//      /***
//       Retreive the resynthesized source at the given index (so long as resyntheiss has happened, mind
//       ***/
//      FluidTensorView<const double, 1> source(const size_t idx) const
//      {
//        assert(idx < mAudioBuffers.rows() && "Range Error");
//        return mAudioBuffers.row(idx);
//      }
//
//      //        source_iterator sources_begin() const ;
//      //        source_iterator sources_end()const;
//
//      /***
//       Get the whole of dictionaries / activations as a 2D structure
//       ***/
//      FluidTensor<double,2> dictionaries() const
//      {
//        return mModel.getW();
//      }
//      FluidTensor<double,2> activations() const
//      {
//        return mModel.getH();
//      }
    private:
      FluidSource<double> mSource;
      FluidSink<double> mSinkResynth;
      FluidSink<double> mSinkDictionaries;
      FluidSink<double> mSinkActivations;
      ProcessModel mArguments;
      
//      size_t mRank;
//      size_t mIterations;
//      size_t mWindowSize;
//      size_t mFFTSize;
//      size_t mHopSize;
//      bool mHasProcessed;
//      bool mHasResynthed;
      fluid::nmf::NMFModel mModel;
      FluidTensor<double,2> mAudioBuffers;
    };
    
    

    
  } //namespace max
} //namesapce fluid
