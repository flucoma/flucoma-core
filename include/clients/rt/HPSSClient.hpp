#pragma once

#include "BaseAudioClient.hpp"
#include "clients/common/STFTCheckParams.hpp"
#include "algorithms/RTHPSS.hpp"
#include "algorithms/STFT.hpp"
#include "clients/common/FluidParams.hpp"

#include <complex>
#include <string>
#include <tuple>


namespace fluid{
namespace hpss{
  template <typename T, typename U>
  class HPSSClient:public audio::BaseAudioClient<T,U>
  {
    static const std::vector<parameter::Descriptor> &getParamDescriptors()
    {
      static std::vector<parameter::Descriptor> params;
      if(params.size() == 0)
      {
        params.emplace_back("psize","Percussive Filter Size",parameter::Type::Long);
        params.back().setMin(3).setDefault(31);
        
        params.emplace_back("hsize","Harmonic Filter Size",parameter::Type::Long);
        params.back().setMin(3).setDefault(17);
        
        audio::BaseAudioClient<T,U>::initParamDescriptors(params);
        params.emplace_back("fftsize","FFT Size", parameter::Type::Long);
        params.back().setMin(-1).setDefault(-1);
      }
      
      return params;
    }
    
    using data_type = FluidTensorView<T,2>;
    using complex   = FluidTensorView<std::complex<T>,1>;
  public:
    HPSSClient() = default;
    HPSSClient(HPSSClient&) = delete;
    HPSSClient operator=(HPSSClient&) = delete;
    
    HPSSClient(size_t maxWindowSize):
    //stft::STFTCheckParams(windowsize,hopsize,fftsize),
    audio::BaseAudioClient<T,U>(maxWindowSize,1,2,3)
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
      
      size_t pBins = parameter::lookupParam("psize", getParams()).getLong();
      size_t hBins = parameter::lookupParam("hsize", getParams()).getLong();
      
      size_t nBins = fftsize / 2 + 1;
      
      mHPSS = std::unique_ptr<rthpss::RTHPSS>(new rthpss::RTHPSS(nBins, pBins, hBins));
      
      mNormWindow = mSTFT->window();
      mNormWindow.apply(mISTFT->window(),[](double& x, double& y)
      {
         x *= y;
      });
      
      mSeparatedSpectra.resize(fftsize/2+1,2);
      mHarms.resize(fftsize/2+1);
      mPerc.resize(fftsize/2+1);
      audio::BaseAudioClient<T,U>::reset();
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
      
      std::tuple<bool, std::string> windowCheck = audio::BaseAudioClient<T,U>::sanityCheck();
      if(!std::get<0>(windowCheck))
      {
        return windowCheck;
      }
      
      parameter::Instance& winSize = parameter::lookupParam("winsize", getParams());
      parameter::Instance& hopSize = parameter::lookupParam("hopsize", getParams());
      parameter::Instance& fftSize = parameter::lookupParam("fftsize", getParams());
      
      return parameter::checkFFTArguments(winSize, hopSize, fftSize);
      
      size_t pSize = parameter::lookupParam("psize", getParams()).getLong();
      size_t hSize = parameter::lookupParam("hsize", getParams()).getLong();
      
      if(pSize > (fftSize.getLong() / 2) + 1 )
      {
        return {false,"Percussive filter can not be bigger than fft size / 2 + 1, and should really be smaller"};
      }
      
      if(!((pSize % 2) && (hSize % 2)))
      {
        return {false, "Both filters must be of odd-numbered length"};
      }
      
      return {true,"Groovy"};
    }
    
    //Here we do an STFT and its inverse
    void process(data_type input, data_type output) override
    {
      complex spec  = mSTFT->processFrame(input.row(0));
      
      mHPSS->processFrame(spec, mSeparatedSpectra);
      
      mHarms = mSeparatedSpectra.col(0);
      mPerc = mSeparatedSpectra.col(1);
//      mSeparatedSpectra.row(0) = spec;
//      mSeparatedSpectra.row(1) = spec;
      
      output.row(0) = mISTFT->processFrame(mHarms);
      output.row(1) = mISTFT->processFrame(mPerc);
      output.row(2) = mNormWindow;
    }
    //Here we gain compensate for the OLA
    void post_process(data_type output) override
    {
      output.row(0).apply(output.row(2),[](double& x, double g){
        if(x)
        {
          x /= g ? g : 1;
        }
      });
      output.row(1).apply(output.row(2),[](double& x, double g){
        if(x)
        {
          x /= g ? g : 1;
        }
      });
    }
    
    std::vector<parameter::Instance>& getParams() override
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
    std::unique_ptr<rthpss::RTHPSS> mHPSS;
    std::unique_ptr<stft::ISTFT> mISTFT;
    FluidTensor<T,1> mNormWindow;
    FluidTensor<std::complex<T>,2> mSeparatedSpectra;
    FluidTensor<std::complex<T>,1> mHarms;
    FluidTensor<std::complex<T>,1> mPerc;
    std::vector<parameter::Instance> mParams;
  };


}//namespace hpss
}//namespace fluid
