#pragma once

#include "algorithms/NMF.hpp"
#include "algorithms/STFT.hpp"
#include "BaseAudioClient.hpp"
#include "clients/common/FluidParams.hpp"
#include "clients/common/STFTCheckParams.hpp"
#include <complex>
#include <string>
#include <tuple>


namespace fluid{
  namespace client{
    template <typename T, typename U>
    class NMFMatch:public client::BaseAudioClient<T,U>
    {
      using data_type = FluidTensorView<T,2>;
      using complex   = FluidTensorView<std::complex<T>,1>;
      using BufferPointer = std::unique_ptr<client::BufferAdaptor::Access>;
    public:
      static const std::vector<client::Descriptor> &getParamDescriptors()
      {
        static std::vector<client::Descriptor> params;
        if(params.size() == 0)
        {

          params.emplace_back("filterbuf", "Filters Buffer",
                              client::Type::kBuffer);
          params.back().setInstantiation(false);

          params.emplace_back("rank", "Rank", client::Type::kLong);
          params.back().setMin(1).setDefault(1).setInstantiation(true);

          params.emplace_back("iterations", "Iterations",
                              client::Type::kLong);
          params.back().setInstantiation(false).setMin(1).setDefault(10).setInstantiation(false);

          params.emplace_back("winsize", "Window Size", client::Type::kLong);
          params.back().setMin(4).setDefault(1024).setInstantiation(true);

          params.emplace_back("hopsize", "Hop Size", client::Type::kLong);
          params.back().setMin(1).setDefault(256).setInstantiation(true);

          params.emplace_back("fftsize", "FFT Size", client::Type::kLong);
          params.back().setMin(-1).setDefault(-1).setInstantiation(true);

        }
        
        return params;
      }
      
      
      
      NMFMatch() = default;
      NMFMatch(NMFMatch&) = delete;
      NMFMatch operator=(NMFMatch&) = delete;
      
      NMFMatch(size_t maxWindowSize):
      client::BaseAudioClient<T,U>(maxWindowSize,1,0,0)
      {
        newParamSet();
      }
      
      void reset()
      {
        size_t winsize = client::lookupParam("winsize", mParams).getLong();
        size_t hopsize = client::lookupParam("hopsize", mParams).getLong();
        size_t fftsize = client::lookupParam("fftsize", mParams).getLong();
        
        mSTFT  = std::unique_ptr<algorithm::STFT> (new algorithm::STFT(winsize,fftsize,hopsize));
        
        mRank = client::lookupParam("rank", getParams()).getLong();
        size_t iterations = client::lookupParam("iterations", getParams()).getLong();
        mNMF = std::unique_ptr<algorithm::NMF>(new algorithm::NMF(mRank,iterations));
//        outbuf = client::lookupParam("outbuf", getParams()).getBuffer();
        filbuf = client::lookupParam("filterbuf", getParams()).getBuffer();
        
//        client::BufferAdaptor::Access outputBuffer(outbuf);
        
//        outputBuffer.resize(mRank,1,1);
        
        client::BufferAdaptor::Access filterBuffer(filbuf);

        tmpFilt.resize(filterBuffer.numFrames(), filterBuffer.numChans());
        tmpOut.resize(mRank);
        
        tmpMagnitude.resize(fftsize / 2 + 1);
        
        client::BaseAudioClient<T,U>::reset(1,mRank,mRank);
      }
      
      std::tuple<bool,std::string> sanityCheck()
      {
        const std::vector<client::Descriptor>& desc = getParamDescriptors();
        //First, let's make sure that we have a complete of parameters of the right sort
        bool sensible = std::equal(mParams.begin(), mParams.end(),desc.begin(),
         [](const client::Instance& i, const client::Descriptor& d)
         {
           return i.getDescriptor() == d;
         });
        
        if(! sensible || (desc.size() != mParams.size()))
        {
          return {false, "Invalid params passed. Were these generated with newParameterSet()?" };
        }
        //Now scan everything for range, until we hit a problem
        //TODO Factor into client::instance
        for(auto&& p: mParams)
        {
          client::Descriptor d = p.getDescriptor();
          bool rangeOk;
          client::Instance::RangeErrorType errorType;
          std::tie(rangeOk, errorType) = p.checkRange();
          if (!rangeOk)
          {
            std::ostringstream msg;
            msg << "Parameter " << d.getName();
            switch (errorType)
            {
            case client::Instance::RangeErrorType::kMin:
              msg << " value below minimum(" << d.getMin() << ")";
              break;
            case client::Instance::RangeErrorType::kMax:
              msg << " value above maximum(" << d.getMin() << ")";
            default:
              assert(false && "This should be unreachable");
            }
            return { false, msg.str()};
          }
        }
        
        std::tuple<bool, std::string> windowCheck = client::BaseAudioClient<T,U>::sanityCheck();
        if(!std::get<0>(windowCheck))
        {
          return windowCheck;
        }
        
        client::Instance& winSize = client::lookupParam("winsize", getParams());
        client::Instance& hopSize = client::lookupParam("hopsize", getParams());
        client::Instance& fftSize = client::lookupParam("fftsize", getParams());
        
        
        
        std::tuple<bool, std::string> fftok =  client::checkFFTArguments(winSize, hopSize, fftSize);
        if(! std::get<0>(fftok))
        {
          return {false, std::get<1>(fftok)};
        }
        
        
        client::BufferAdaptor::Access filters(client::lookupParam("filterbuf", getParams()).getBuffer());
//        client::BufferAdaptor::Access output(client::lookupParam("outbuf", getParams()).getBuffer());
        
        size_t rank = client::lookupParam("rank", getParams()).getLong();
        size_t iterations = client::lookupParam("iterations", getParams()).getLong();
        
        if(!filters.valid())
        {
          return {false, "Filters buffer invalid"};
        }
//        
//        if(!output.valid())
//        {
//          return {false, "Output buffer invalid"};
//        }
        
        if(filters.numChans() != rank || filters.numFrames() != ((fftSize.getLong() / 2) + 1))
        {
          return {false, "Filters buffer needs to be (fftsize / 2 + 1) frames by rank channels"};
        }
        return {true,"Groovy"};
      }
      
      void process(data_type input, data_type output) override
      {        
        if(filbuf)
        {
          
          client::BufferAdaptor::Access filterBuffer(filbuf);
//          client::BufferAdaptor::Access outputBuffer(outbuf);
          
          if(!filterBuffer.valid())
          {
            return;
          }
          
          
          for(size_t i = 0; i < tmpFilt.cols(); ++i)
            tmpFilt.col(i) = filterBuffer.samps(0,i);
          
          tmpMagnitude.apply(mSTFT->processFrame(input.row(0)), [](double& x, std::complex<double>& y)->double{
            x = std::abs(y);
          });
          
          mNMF->processFrame(tmpMagnitude, tmpFilt, tmpOut);
          
          double hsum = std::accumulate(tmpOut.begin(), tmpOut.end(), 0.0);
//          std::transform(tmpOut.begin(), tmpOut.end(), tmpOut.begin(), [&](double& x)->double{
//            return hsum? x / hsum : 0;
//          });
          
          output.col(0) = tmpOut; 
   
        }
      }
      
      //Here we gain compensate for the OLA
      void postProcess(data_type output) override {}

      std::vector<client::Instance>& getParams() override
      {
        return mParams;
      }
      
//      size_t getWindowSize() override
//      {
////        assert(mExtractor);
////        return mExtractor->inputSize();
//      }
//      
//      size_t getHopSize() override
//      {
////        assert(mExtractor);
////        return mExtractor->hopSize();
//      }
//      
    private:
      void newParamSet()
      {
        mParams.clear();
        for(auto&& d: getParamDescriptors())
          mParams.emplace_back(d);
      }
      
      
      std::unique_ptr<algorithm::STFT> mSTFT; 
      std::unique_ptr<algorithm::NMF> mNMF;
      client::BufferAdaptor* outbuf = nullptr;
      client::BufferAdaptor* filbuf = nullptr;
      FluidTensor<double, 2> tmpFilt;
      FluidTensor<double, 1> tmpMagnitude;
      FluidTensor<double, 1> tmpOut;
      size_t mRank;
//      BufferPointer filterBuffer;
//      BufferPointer outputBuffer;
      
//      std::unique_ptr<TransientSegmentation> mExtractor;
//      FluidTensor<std::complex<T>,1> mTransients;
      std::vector<client::Instance> mParams;
    };
  }//namespace client
}//namespace fluid

