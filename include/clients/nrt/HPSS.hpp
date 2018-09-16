#pragma once

#include "clients/common/FluidParams.hpp"
#include "clients/common/STFTCheckParams.hpp"
#include "data/FluidTensor.hpp"
#include "data/FluidBuffers.hpp"
#include "algorithms/RTHPSS.hpp"
#include "algorithms/STFT.hpp"
#include "algorithms/RatioMask.hpp"


#include <string>
#include <sstream> //for ostringstream
#include <utility> //for std make_pair
#include <unordered_set>
#include <vector> //for containers of params, and for checking things

namespace fluid {
  namespace hpss{

    /**
     Integration class for doing NMF filtering and resynthesis
     **/
    class HPSSClient
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
        size_t harmsize;
        size_t percsize;
        parameter::BufferAdaptor* src = 0;
        parameter::BufferAdaptor* harm = 0;
        parameter::BufferAdaptor* perc = 0;
        parameter::BufferAdaptor* res = 0;
        
        double pThreshFreq[2];
        double pThreshAmp[2];
        double hThreshFreq[2];
        double hThreshAmp[2];
        
        bool pbinary;
        bool hbinary;
        
        
        size_t winsize;
        size_t hopsize;
        size_t fftsize;
      };

      static const std::vector<parameter::Descriptor>& getParamDescriptors()
      {
        static std::vector<desc_type> params;
        if(params.empty())
        {
          params.emplace_back("src","First Source Buffer", parameter::Type::Buffer);
          params.back().setInstantiation(true);

          params.emplace_back("offsetframes","Source Offset", parameter::Type::Long);
          params.back().setInstantiation(true).setMin(0).setDefault(0);

          params.emplace_back("numframes","Source Frames", parameter::Type::Long);
          params.back().setInstantiation(true).setMin(-1).setDefault(-1);

          params.emplace_back("offsetchans","Source Channel Offset", parameter::Type::Long);
          params.back().setInstantiation(true).setMin(0).setDefault(0);

          params.emplace_back("numchans","Source Channels", parameter::Type::Long);
          params.back().setInstantiation(true).setMin(-1).setDefault(-1);

          params.emplace_back("harmbuf","Harmonic Component Buffer", parameter::Type::Buffer);
          params.back().setInstantiation(false);
          
          params.emplace_back("percbuf","Percussive Component Buffer", parameter::Type::Buffer);
          params.back().setInstantiation(false);
          
          params.emplace_back("resbuf", "Residual Component Buffer", parameter::Type::Buffer);
          params.back().setInstantiation(false);
          
          params.emplace_back("psize","Percussive Filter Size",parameter::Type::Long);
          params.back().setMin(3).setDefault(17).setInstantiation(false);
          
          params.emplace_back("hsize","Harmonic Filter Size",parameter::Type::Long);
          params.back().setMin(3).setDefault(17).setInstantiation(false);
          
          params.emplace_back("binaryp","Percussive Binary Mask", parameter::Type::Long);
          params.back().setMin(0).setMax(1).setInstantiation(false).setDefault(0);
          
          params.emplace_back("binaryh","Harmonic Binary Mask", parameter::Type::Long);
          params.back().setMin(0).setMax(1).setInstantiation(false).setDefault(0);
          
          params.emplace_back("pthreshf1","Percussive Threshold Low Frequency ",parameter::Type::Float);
          params.back().setMin(0).setMax(1).setDefault(0).setInstantiation(false);
          
          params.emplace_back("pthresha1","Percussive Threshold Low Amplitude",parameter::Type::Float);
          params.back().setDefault(0).setInstantiation(false);
          
          params.emplace_back("pthreshf2","Percussive Threshold High Frequency",parameter::Type::Float);
          params.back().setMin(0).setMax(1).setDefault(1).setInstantiation(false);
          
          params.emplace_back("pthresha2","Percussive Threshold High Amplitude",parameter::Type::Float);
          params.back().setDefault(0).setInstantiation(false);
          
          params.emplace_back("hthreshf1","Harmonic Threshold Low Frequency",parameter::Type::Float);
          params.back().setMin(0).setMax(1).setDefault(0).setInstantiation(false);
          
          params.emplace_back("hthresha1","Harmonic Threshold Low Amplitude",parameter::Type::Float);
          params.back().setDefault(0).setInstantiation(false);
          
          params.emplace_back("hthreshf2","Harmonic Threshold High Frequency",parameter::Type::Float);
          params.back().setMin(0).setMax(1).setDefault(1).setInstantiation(false);
          
          params.emplace_back("hthresha2","Harmonic Threshold High Amplitude",parameter::Type::Float);
          params.back().setDefault(0).setInstantiation(false);
          
          params.emplace_back(desc_type{"winsize","Window Size", parameter::Type::Long});
          params.back().setMin(4).setDefault(4096);
          
          params.emplace_back(desc_type{"hopsize","Hop Size", parameter::Type::Long});
          params.back().setMin(1).setDefault(1024);
          
          params.emplace_back(desc_type{"fftsize","FFT Size", parameter::Type::Long});
          params.back().setMin(-1).setDefault(-1);
          
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

        if(bufCount < 3)
        {
          return { false, "Expecting three valid buffers", model};
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
        model.src           = parameter::lookupParam("src",getParams()).getBuffer();
        model.offset        = srcOffset;
        model.frames        = srcFrames > 0 ? srcFrames : src.numFrames() - model.offset;
        model.channelOffset = srcChanOffset;
        model.channels      = srcChans >  0 ? srcChans  : src.numChans() - model.channelOffset;


        parameter::BufferAdaptor::Access harmBuf(parameter::lookupParam("harmbuf",getParams()).getBuffer());
        parameter::BufferAdaptor::Access percBuf(parameter::lookupParam("percbuf", getParams()).getBuffer());

        
        if(! harmBuf.valid())
        {
          return {false,"Harmonic buffer invalid",model};
        }
        if(! percBuf.valid())
        {
          return {false,"Percussive buffer invalid",model};
        }
        
        

        model.harm = parameter::lookupParam("harmbuf",getParams()).getBuffer();
        model.perc = parameter::lookupParam("percbuf", getParams()).getBuffer();
        
        parameter::Instance& winSize = parameter::lookupParam("winsize", getParams());
        parameter::Instance& hopSize = parameter::lookupParam("hopsize", getParams());
        parameter::Instance& fftSize = parameter::lookupParam("fftsize", getParams());
        
        
        std::tuple<bool,std::string> fftok = parameter::checkFFTArguments(winSize, hopSize, fftSize);
        if(!std::get<0>(fftok))
        {
          return std::tuple_cat(fftok,std::make_tuple(model));
        }
        
        size_t pSize = parameter::lookupParam("psize", getParams()).getLong();
        size_t hSize = parameter::lookupParam("hsize", getParams()).getLong();
        
        if(pSize > (fftSize.getLong() / 2) + 1 )
        {
          return {false,"Percussive filter can not be bigger than fft size / 2 + 1, and should really be smaller",model};
        }
        
        if(!((pSize % 2) && (hSize % 2)))
        {
          return {false, "Both filters must be of odd-numbered length",model};
        }
        
        model.harmsize = hSize;
        model.percsize = pSize;
        model.winsize = winSize.getLong();
        model.hopsize = hopSize.getLong();
        model.fftsize = fftSize.getLong();
        
        model.hbinary = parameter::lookupParam("binaryh", getParams()).getLong() > 0;
        model.pbinary = parameter::lookupParam("binaryp", getParams()).getLong() > 0;
        
        if(model.hbinary || model.pbinary)
        {
          parameter::BufferAdaptor::Access resBuf(parameter::lookupParam("resbuf",getParams()).getBuffer());

          if(!resBuf.valid())
          {
            return {false, "One or both masks is binary but no residual output buffer supplied",model};
          }
          model.res = parameter::lookupParam("resbuf",getParams()).getBuffer();
          
          double pf1 = parameter::lookupParam("pthreshf1", getParams()).getFloat();
          double pf2 = parameter::lookupParam("pthreshf2", getParams()).getFloat();
          if(pf1 >= pf2)
          {
            return {false,"Percussive Threshold low frequency must be below high frequency",model};
          }
          
          double pa1 = parameter::lookupParam("pthresha1", getParams()).getFloat();
          double pa2 = parameter::lookupParam("pthresha2", getParams()).getFloat();
          model.pThreshFreq[0] = pf1;
          model.pThreshFreq[1] = pf2;
          model.pThreshAmp[0] =  pa1;
          model.pThreshAmp[0] = pa2;
          
          double hf1 = parameter::lookupParam("hthreshf1", getParams()).getFloat();
          double hf2 = parameter::lookupParam("hthreshf2", getParams()).getFloat();
          if(hf1 >= hf2)
          {
            return {false,"Harmonic Threshold low frequency must be below high frequency",model};
          }
          
          double ha1 = parameter::lookupParam("hthresha1", getParams()).getFloat();
          double ha2 = parameter::lookupParam("hthresha2", getParams()).getFloat();
          model.hThreshFreq[0] = hf1;
          model.hThreshFreq[1] = hf2;
          model.hThreshAmp[0] = ha1;
          model.hThreshAmp[0] = ha2;
        }
        //We made it
        return {true, "Everything is lovely",model};
      }


      //No, you may not  copy this, or move this
      HPSSClient(HPSSClient&)=delete;
      HPSSClient(HPSSClient&&)=delete;
      HPSSClient operator=(HPSSClient&)=delete;
      HPSSClient operator=(HPSSClient&&)=delete;

      HPSSClient(){
        newParameterSet();
      }

      void process(ProcessModel model)
      {
        parameter::BufferAdaptor::Access src(model.src);
        parameter::BufferAdaptor::Access harm(model.harm);
        parameter::BufferAdaptor::Access perc(model.perc);
        parameter::BufferAdaptor::Access res(model.res);
        
        harm.resize(model.frames,model.channels,1);
        perc.resize(model.frames,model.channels,1);
        
        if(model.hbinary || model.pbinary)
          res.resize(model.frames,model.channels,1);
          
        stft::STFT  stft (model.winsize,model.fftsize,model.hopsize);
        stft::ISTFT istft(model.winsize,model.fftsize,model.hopsize);
 
        rthpss::RTHPSS processor(model.fftsize / 2 + 1, model.percsize,model.harmsize,model.hbinary, model.pbinary,
                                 model.hThreshFreq[0], model.hThreshAmp[0],
                                 model.hThreshFreq[1], model.hThreshAmp[1],
                                 model.pThreshFreq[0], model.pThreshAmp[0],
                                 model.pThreshFreq[1], model.pThreshAmp[1]);

        
        size_t lag = model.hopsize * (model.harmsize - 1);
        for(size_t i = 0; i < model.channels;++i)
        {
          FluidTensor<double,1> input(model.frames + lag);
          
          input.fill(0); 
          input(fluid::slice(0,model.frames)) =  (src.samps(model.offset,model.frames,model.channelOffset + i ,1).col(0));
          
          
//          double max = *std::max_element(input.begin(), input.end());
//          std::cout << "MAX " << max << '\n';
//          
//          auto n = std::find_if(input.begin(),input.end(),[](double d){return std::isnan(d);});
//          auto idx = std::distance(input.begin(), n);
//          assert((n == input.end()));
          
          
          auto spectrum = stft.process(input);
          
//          auto isanan = std::find_if(spectrum.mData.begin(),spectrum.mData.end(),[](std::complex<double> d){
//            return (std::isnan(d.real()) || std::isnan(d.imag()));
//          });
//
//          idx = std::distance(spectrum.mData.begin(), isanan);
//          assert(isanan == spectrum.mData.end());
          
          
          
          
          FluidTensor<std::complex<double>,2> harmonicSpec(spectrum.mData.rows(),spectrum.mData.cols());
          FluidTensor<std::complex<double>,2> percussiveSpec(spectrum.mData.rows(),spectrum.mData.cols());;
          FluidTensor<std::complex<double>,2> residualSpec(spectrum.mData.rows(),spectrum.mData.cols());;
          FluidTensor<std::complex<double>,2> result(model.fftsize / 2 + 1, 3);
          for(size_t j = 0; j < spectrum.mData.rows(); ++j)
          {
            processor.processFrame(spectrum.mData.row(j), result);
            harmonicSpec.row(j)   = result.col(0);
            percussiveSpec.row(j) = result.col(1);
            residualSpec.row(j)   = result.col(2);
          }
          auto harmonicAudio = istft.process(harmonicSpec);
          auto percussiveAudio = istft.process(percussiveSpec);
          
          
          
          harm.samps(i,0) = harmonicAudio  (fluid::slice(lag,model.frames));
          perc.samps(i,0) = percussiveAudio(fluid::slice(lag,model.frames));
          
          if(model.hbinary || model.pbinary)
          {
            auto residualAudio = istft.process(residualSpec);
            res.samps(i,0) = residualAudio(fluid::slice(lag,model.frames));
          }
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
