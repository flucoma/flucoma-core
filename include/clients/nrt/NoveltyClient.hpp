#pragma once

#include "clients/common/FluidParams.hpp"
#include "clients/common/STFTCheckParams.hpp"
#include "data/FluidTensor.hpp"
#include "data/FluidBuffers.hpp"
#include "algorithms/NoveltySegmentation.hpp"
#include "algorithms/STFT.hpp"



#include <string>
#include <sstream> //for ostringstream
#include <utility> //for std make_pair
#include <unordered_set>
#include <vector> //for containers of params, and for checking things

namespace fluid {
  namespace segmentation{

    /**
     Integration class for doing NMF filtering and resynthesis
     **/
      class NoveltyClient
    {
      using desc_type = parameter::Descriptor;
      using param_type = parameter::Instance;
    public:

      struct ProcessModel
      {
        size_t frames;
        size_t offset;
        size_t channels;
        size_t channelOffset;
        parameter::BufferAdaptor* src = 0;
        parameter::BufferAdaptor* indices = 0;
        
        size_t kernelsize;
        double threshold;
        size_t filtersize;
        
        size_t winsize;
        size_t hopsize;
        size_t fftsize;
      };

      
      /***
       int bandwidth,
       double threshold, int minTrackLength, double magWeight,
       double freqWeight
       
       //
       // * bandwidth - width in bins of the fragment of window transform correlated with each frame
       // should have an effect on cost vs quality
       // * threshold (0 to 1) select more or less peaks as sinusoidal from the normalized cross-correlation
       // * min length (frames): minimum length of a sinusoidal track (0 for no tracking)
       // * weight of spectral magnitude when associating a peak to an existing track (relative, but suggested 0 to 1)
       // * weight of frequency when associating a peak to an existing track (relativer, suggested 0 to 1)
       ***/
      static const std::vector<parameter::Descriptor>& getParamDescriptors()
      {
        static std::vector<desc_type> params;
        if(params.empty())
        {
          params.emplace_back("src","Source Buffer", parameter::Type::Buffer);
          params.back().setInstantiation(true);

          params.emplace_back("offsetframes","Source Offset", parameter::Type::Long);
          params.back().setInstantiation(true).setMin(0).setDefault(0);

          params.emplace_back("numframes","Source Frames", parameter::Type::Long);
          params.back().setInstantiation(true).setMin(-1).setDefault(-1);

          params.emplace_back("offsetchans","Source Channel Offset", parameter::Type::Long);
          params.back().setInstantiation(true).setMin(0).setDefault(0);

          params.emplace_back("numchans","Source Channels", parameter::Type::Long);
          params.back().setInstantiation(true).setMin(-1).setDefault(-1);

          params.emplace_back(desc_type{"transbuf","Indices Buffer", parameter::Type::Buffer});
          params.back().setInstantiation(false);
      
          params.emplace_back(desc_type{"kernelsize","Kernel Size", parameter::Type::Long});
          params.back().setMin(3).setDefault(3).setInstantiation(false);
          
          params.emplace_back(desc_type{"threshold","Threshold", parameter::Type::Float});
          params.back().setMin(0).setDefault(0.8).setInstantiation(false);
          
          params.emplace_back("filtersize","Smoothing Filter Size", parameter::Type::Long);
          params.back().setInstantiation(false);
          
          
          params.emplace_back(desc_type{"winsize","Window Size", parameter::Type::Long});
          params.back().setMin(4).setDefault(1024).setInstantiation(false);
          
          params.emplace_back(desc_type{"hopsize","Hop Size", parameter::Type::Long});
          params.back().setMin(1).setDefault(512).setInstantiation(false);
          
          params.emplace_back(desc_type{"fftsize","FFT Size", parameter::Type::Long});
          params.back().setMin(-1).setDefault(2048).setInstantiation(false);
          
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
        parameter::BufferAdaptor::Access src(mParams[0].getBuffer());

        if(!src.valid())
        {
          return  { false, "Source buffer doesn't exist or can't be accessed.", model };
        }

        if(src.numFrames() == 0)
        {
          return {false, "Source buffer is empty",model};
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
          return { false, "Expecting two valid buffers", model};
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

        long srcOffset     = parameter::lookupParam("offsetframes", mParams).getLong();
        long srcFrames     = parameter::lookupParam("numframes",    mParams).getLong();
        long srcChanOffset = parameter::lookupParam("offsetchans",  mParams).getLong();
        long srcChans      = parameter::lookupParam("numchans",     mParams).getLong();
        
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
        model.src          = parameter::lookupParam("src",getParams()).getBuffer();
        model.offset        = srcOffset;
        model.frames        = srcFrames > 0 ? srcFrames : src.numFrames() - model.offset;
        model.channelOffset = srcChanOffset;
        model.channels      = srcChans >  0 ? srcChans  : src.numChans() - model.channelOffset;

        model.indices = parameter::lookupParam("transbuf",getParams()).getBuffer();

        parameter::Instance& winSize = parameter::lookupParam("winsize", getParams());
        parameter::Instance& hopSize = parameter::lookupParam("hopsize", getParams());
        parameter::Instance& fftSize = parameter::lookupParam("fftsize", getParams());
        
        
        std::tuple<bool,std::string> fftok = parameter::checkFFTArguments(winSize, hopSize, fftSize);
        if(!std::get<0>(fftok))
        {
          return std::tuple_cat(fftok,std::make_tuple(model));
        }
        
        
        size_t kernelSize = parameter::lookupParam("kernelsize",getParams()).getLong();
        
        if(!(kernelSize % 2))
        {
          return {false, "Kernel size must be odd", model};
        }
        
        
        if(kernelSize * hopSize.getLong() > model.frames)
        {
          return {false,"Kernel size is bigger than number of available samples", model};
        }
        
        size_t filterSize = parameter::lookupParam("filtersize", getParams()).getLong();
        
        
        
        model.kernelsize = kernelSize;
        model.filtersize = filterSize;
        model.winsize = winSize.getLong();
        model.hopsize = hopSize.getLong();
        model.fftsize = fftSize.getLong();
        model.threshold =  parameter::lookupParam("threshold",getParams()).getFloat();

        //We made it
        return {true, "Everything is lovely",model};
      }


      //No, you may not  copy this, or move this
      NoveltyClient(NoveltyClient&)=delete;
      NoveltyClient(NoveltyClient&&)=delete;
      NoveltyClient operator=(NoveltyClient&)=delete;
      NoveltyClient operator=(NoveltyClient&&)=delete;

      NoveltyClient(){
        newParameterSet();
      }

      void process(ProcessModel model)
      {
        parameter::BufferAdaptor::Access src(model.src);
        parameter::BufferAdaptor::Access idx(model.indices);
        
        
        FluidTensor<double,1> monoSource(model.frames);
        
        
        //Make a mono sum;
        for(size_t i = 0; i < model.channels; ++i)
        {
          monoSource.apply(src.samps(model.offset, model.frames, model.channelOffset + i), [](double& x, double y){
            x += y;
          });
        }
        
        stft::STFT stft(model.winsize,model.fftsize,model.hopsize);
        stft::ISTFT istft(model.winsize,model.fftsize,model.hopsize);
        
        novelty::NoveltySegmentation processor(model.kernelsize,model.threshold,model.filtersize);
        
        auto  spectrum = stft.process(monoSource);
        auto magnitude = spectrum.getMagnitude();
        
        FluidTensor<double,1> changePoints(magnitude.rows());
        
        processor.process(magnitude,changePoints);
        
        size_t num_spikes = std::accumulate(changePoints.begin(), changePoints.end() , 0);
        
        //Arg sort
        std::vector<size_t> indices(changePoints.size());
        std::iota(indices.begin(),indices.end(),0);
        std::sort(indices.begin(), indices.end(),[&](size_t i1, size_t i2){
          return changePoints[i1] > changePoints[i2];
        });
        
        //Now put the gathered indicies into ascending order
        std::sort(indices.begin(), indices.begin() + num_spikes);
        
        //convert frame numbers to samples, and account for any offset
        std::transform(indices.begin(),indices.begin() + num_spikes, indices.begin(), [&](size_t x)->size_t {
          x *= model.hopsize;
          x += model.offset;
          return x; 
        });
        
        //Place a leading <offset> and <numframes>
        indices.insert(indices.begin() + num_spikes, model.offset + model.frames);
        indices.insert(indices.begin(), model.offset);
        
        idx.resize(num_spikes + 2,1,1);
        
    
        idx.samps(0) = FluidTensorView<size_t,1>{indices.data(),0,num_spikes + 2};
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
