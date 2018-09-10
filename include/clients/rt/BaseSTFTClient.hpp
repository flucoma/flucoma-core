  /*
 @file: BaseSTFTClient

 Base class for real-time STFT processes

*/
#pragma once
#include "clients/rt/BaseAudioClient.hpp"
#include "clients/common/STFTCheckParams.hpp"
#include "algorithms/STFT.hpp"
#include "clients/common/FluidParams.hpp"

#include <complex>
#include <string>
#include <tuple>


namespace  fluid {
namespace audio {


  template <typename T, typename U>
  class BaseSTFTClient:public BaseAudioClient<T,U>
    {
      static const std::vector<parameter::Descriptor> &getParamDescriptors()
      {
        static std::vector<parameter::Descriptor> params;
        if(params.size() == 0)
        {
          params.emplace_back("winsize","Window Size", parameter::Type::Long);
          params.back().setMin(4).setDefault(1024);
          
          params.emplace_back("hopsize","Hop Size", parameter::Type::Long);
          params.back().setMin(1).setDefault(256);
          
          params.emplace_back("fftsize","FFT Size", parameter::Type::Long);
          params.back().setMin(-1).setDefault(-1);
        }
        
        return params;
      }
        
      using data_type = FluidTensorView<T,2>;
      using complex   = FluidTensorView<std::complex<T>,1>;
    public:
        BaseSTFTClient() = default;
        BaseSTFTClient(BaseSTFTClient&) = delete;
        BaseSTFTClient operator=(BaseSTFTClient&) = delete;

        BaseSTFTClient(size_t maxWindowSize):
        //stft::STFTCheckParams(windowsize,hopsize,fftsize),
        BaseAudioClient<T,U>(maxWindowSize,1,1,2 )
        {
          newParamSet();
        }

      void reset() override
      {
        size_t winsize = parameter::lookupParam("winsize", mParams).getLong();
        size_t hopsize = parameter::lookupParam("hopsize", mParams).getLong();
        size_t fftsize = parameter::lookupParam("fftsize", mParams).getLong();
        
        mSTFT  = std::unique_ptr<stft::STFT> (new stft::STFT(winsize,fftsize,hopsize));
        mISTFT = std::unique_ptr<stft::ISTFT>(new stft::ISTFT(winsize,fftsize,hopsize));
        
        normWindow = mSTFT->window();
        normWindow.apply(mISTFT->window(),[](double& x, double& y)
        {
          x *= y;
        });
        

        BaseAudioClient<T,U>::reset();
      }
      
      std::tuple<bool,std::string> sanityCheck()
      {
//        BaseAudioClient<T,U>::getParams()[0].setLong(parameter::lookupParam("winsize", mParams).getLong());
//        BaseAudioClient<T,U>::getParams()[1].setLong(parameter::lookupParam("hopsize", mParams).getLong());
        
        for(auto&& p: getParams())
        {
          std::tuple<bool,parameter::Instance::RangeErrorType> res = p.checkRange();
          if(!std::get<0>(res))
          {
            switch(std::get<1>(res))
            {
              case parameter::Instance::RangeErrorType::Min:
                return {false,"Parameter below minimum"};
                break;
              case parameter::Instance::RangeErrorType::Max:
                return {false,"Parameter above maximum"};
                break;
            }
          }
          
        }
        
        
        std::tuple<bool, std::string> windowCheck =  BaseAudioClient<T,U>::sanityCheck();
        if(!std::get<0>(windowCheck))
        {
          return windowCheck;
        }
        
        return parameter::checkFFTArguments(mParams[0], mParams[1], mParams[2]);
      }
      
        //Here we do an STFT and its inverse
        void process(data_type input, data_type output) override
        {
            complex spec  = mSTFT->processFrame(input.row(0));
            output.row(0) = mISTFT->processFrame(spec);
            output.row(1) = normWindow;
        }
        //Here we gain compensate for the OLA
        void post_process(data_type output) override
        {
          output.row(0).apply(output.row(1),[](double& x, double g){
              if(x)
              {
                x /= g ? g : 1;
              }
          });
        }
      
     std::vector<parameter::Instance>& getParams()
      {
        return mParams;
      }
      
    private:
      void newParamSet()
      {
        mParams.clear();
        for(auto&& d: getParamDescriptors())
          mParams.emplace_back(d);
      }
      
      std::unique_ptr<stft::STFT> mSTFT;
      std::unique_ptr<stft::ISTFT> mISTFT;
      FluidTensor<T,1> normWindow;
      std::vector<parameter::Instance> mParams;
    };
} //namespace audio
}//namespace fluid
