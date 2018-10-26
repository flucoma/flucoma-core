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

    
    using data_type = FluidTensorView<T,2>;
    using complex   = FluidTensorView<std::complex<T>,1>;
  public:
    
    static const std::vector<parameter::Descriptor> &getParamDescriptors()
    {
      static std::vector<parameter::Descriptor> params;
      if(params.size() == 0)
      {

        params.emplace_back("hsize", "Harmonic Filter Size",
                            parameter::Type::kLong);
        params.back().setMin(3).setDefault(17).setInstantiation(true);

        params.emplace_back("psize", "Percussive Filter Size",
                            parameter::Type::kLong);
        params.back().setMin(3).setDefault(31).setInstantiation(true);

        params.emplace_back("modeflag", "Masking Mode", parameter::Type::kLong);
        params.back().setMin(0).setMax(2).setInstantiation(true).setDefault(0);

        params.emplace_back("htf1", "Harmonic Threshold Low Frequency",
                            parameter::Type::kFloat);
        params.back().setMin(0).setMax(1).setDefault(0).setInstantiation(false);

        params.emplace_back("hta1", "Harmonic Threshold Low Amplitude",
                            parameter::Type::kFloat);
        params.back().setDefault(0).setInstantiation(false);

        params.emplace_back("htf2", "Harmonic Threshold High Frequency",
                            parameter::Type::kFloat);
        params.back().setMin(0).setMax(1).setDefault(1).setInstantiation(false);

        params.emplace_back("hta2", "Harmonic Threshold High Amplitude",
                            parameter::Type::kFloat);
        params.back().setDefault(0).setInstantiation(false);

        params.emplace_back("ptf1", "Percussive Threshold Low Frequency ",
                            parameter::Type::kFloat);
        params.back().setMin(0).setMax(1).setDefault(0).setInstantiation(false);

        params.emplace_back("pta1", "Percussive Threshold Low Amplitude",
                            parameter::Type::kFloat);
        params.back().setDefault(0).setInstantiation(false);

        params.emplace_back("ptf2", "Percussive Threshold High Frequency",
                            parameter::Type::kFloat);
        params.back().setMin(0).setMax(1).setDefault(1).setInstantiation(false);

        params.emplace_back("pta2", "Percussive Threshold High Amplitude",
                            parameter::Type::kFloat);
        params.back().setDefault(0).setInstantiation(false);
        
        audio::BaseAudioClient<T,U>::initParamDescriptors(params);
        params.emplace_back("fftsize", "FFT Size", parameter::Type::kLong);
        params.back().setMin(-1).setDefault(-1).setInstantiation(true);
      }
      return params;
    }
    
    
    HPSSClient() = default;
    HPSSClient(HPSSClient&) = delete;
    HPSSClient operator=(HPSSClient&) = delete;
    
    HPSSClient(size_t maxWindowSize):
    //stft::STFTCheckParams(windowsize,hopsize,fftsize),
    audio::BaseAudioClient<T,U>(maxWindowSize,1,3,4)
    {
      newParamSet();
    }
    
    void reset() 
    {
      size_t winsize = parameter::lookupParam("winsize", mParams).getLong();
      size_t hopsize = parameter::lookupParam("hopsize", mParams).getLong();
      size_t fftsize = parameter::lookupParam("fftsize", mParams).getLong();
      
      mSTFT  = std::unique_ptr<stft::STFT> (new stft::STFT(winsize,fftsize,hopsize));
      mISTFT = std::unique_ptr<stft::ISTFT>(new stft::ISTFT(winsize,fftsize,hopsize));
      
      size_t pBins   = parameter::lookupParam("psize", getParams()).getLong();
      size_t hBins   = parameter::lookupParam("hsize", getParams()).getLong();

      double pthreshF1 = parameter::lookupParam("ptf1", getParams()).getFloat();
      double pthreshA1 = parameter::lookupParam("pta1", getParams()).getFloat();
      double pthreshF2 = parameter::lookupParam("ptf2", getParams()).getFloat();
      double pthreshA2 = parameter::lookupParam("pta2", getParams()).getFloat();

      double hthreshF1 = parameter::lookupParam("htf1", getParams()).getFloat();
      double hthreshA1 = parameter::lookupParam("hta1", getParams()).getFloat();
      double hthreshF2 = parameter::lookupParam("htf2", getParams()).getFloat();
      double hthreshA2 = parameter::lookupParam("hta2", getParams()).getFloat();

      size_t nBins = fftsize / 2 + 1;
      
      mHPSS = std::unique_ptr<rthpss::RTHPSS>(new rthpss::RTHPSS(nBins, pBins, hBins,mode(), hthreshF1, hthreshA1, hthreshF2, hthreshA2, pthreshF1, pthreshA1, pthreshF2, pthreshA2));
                                                                 
      
      mNormWindow = mSTFT->window();
      mNormWindow.apply(mISTFT->window(),[](double& x, double& y)
      {
         x *= y;
      });
      
      mSeparatedSpectra.resize(fftsize/2+1,3);
      mHarms.resize(fftsize/2+1);
      mPerc.resize(fftsize/2+1);
      mResidual.resize(fftsize/2+1);
      audio::BaseAudioClient<T,U>::reset();
    }
    
    std::tuple<bool,std::string> sanityCheck()
    {
      for(auto&& p: getParams())
      {
        std::tuple<bool,parameter::Instance::RangeErrorType> res = p.checkRange();
        if(!std::get<0>(res))
        {
          switch(std::get<1>(res))
          {
          case parameter::Instance::RangeErrorType::kMin:
            return {false, "Parameter below minimum"};
            break;
          case parameter::Instance::RangeErrorType::kMax:
            return {false, "Parameter above maximum"};
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
      
//      mHPSS->setHThreshold(parameter::lookupParam("hthresh", getParams()).getFloat());
//      mHPSS->setPThreshold(parameter::lookupParam("pthresh", getParams()).getFloat());
      
      
      if(mode() > 0)
      {
        mHPSS->setHThresholdX1(parameter::lookupParam("htf1", getParams()).getFloat());
        mHPSS->setHThresholdY1(parameter::lookupParam("hta1", getParams()).getFloat());
        mHPSS->setHThresholdX2(parameter::lookupParam("htf2", getParams()).getFloat());
        mHPSS->setHThresholdY2(parameter::lookupParam("hta2", getParams()).getFloat());
      }
      
      if(mode() == 2)
      {
        mHPSS->setPThresholdX1(parameter::lookupParam("ptf1", getParams()).getFloat());
        mHPSS->setPThresholdY1(parameter::lookupParam("pta1", getParams()).getFloat());
        mHPSS->setPThresholdX2(parameter::lookupParam("ptf2", getParams()).getFloat());
        mHPSS->setPThresholdY2(parameter::lookupParam("pta2", getParams()).getFloat());
      }
     
      
      mHPSS->processFrame(spec, mSeparatedSpectra);
      
      mHarms = mSeparatedSpectra.col(0);
      mPerc  = mSeparatedSpectra.col(1);
//      mSeparatedSpectra.row(0) = spec;
//      mSeparatedSpectra.row(1) = spec;
      
      output.row(0) = mISTFT->processFrame(mHarms);
      output.row(1) = mISTFT->processFrame(mPerc);
      if(mode() == 2)
      {
        mResidual = mSeparatedSpectra.col(2);
        output.row(2) = mISTFT->processFrame(mResidual);
      }
      output.row(3) = mNormWindow;
    }
    //Here we gain compensate for the OLA
    void postProcess(data_type output) override {
      output.row(0).apply(output.row(3),[](double& x, double g){
        if(x)
        {
          x /= g ? g : 1;
        }
      });
      output.row(1).apply(output.row(3),[](double& x, double g){
        if(x)
        {
          x /= g ? g : 1;
        }
      });
      
      if(mode() == 2)
        output.row(2).apply(output.row(3),[](double& x, double g){
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
    
    size_t mode()
    {
      return parameter::lookupParam("modeflag", getParams()).getLong();
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
    FluidTensor<std::complex<T>,1> mResidual;
    std::vector<parameter::Instance> mParams;
  };
}//namespace hpss
}//namespace fluid
