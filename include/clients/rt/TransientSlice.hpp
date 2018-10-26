#pragma once

#include "algorithms/TransientSegmentation.hpp"

#include "BaseAudioClient.hpp"

#include "clients/common/FluidParams.hpp"

#include <complex>
#include <string>
#include <tuple>


namespace fluid{
  namespace segmentation{
    template <typename T, typename U>
    class TransientsSlice:public audio::BaseAudioClient<T,U>
    {
      using data_type = FluidTensorView<T,2>;
      using complex   = FluidTensorView<std::complex<T>,1>;
    public:
      
      static const std::vector<parameter::Descriptor> &getParamDescriptors()
      {
        static std::vector<parameter::Descriptor> params;
        if(params.size() == 0)
        {
          //Determines input / hop size, can't yet set at perform time

          params.emplace_back("order", "Order", parameter::Type::kLong);
          params.back().setInstantiation(false).setMin(20).setDefault(50).setInstantiation(true);
          //order min > paramDetectHalfWindow, or ~40-50 ms,
          
          //          int paramBlockSize = 2048;  // The main block size for processing (higher == longer processing times N^2 but better quality)
          params.emplace_back("blocksize", "Block Size",
                              parameter::Type::kLong);
          params.back().setInstantiation(false).setMin(100).setDefault(256).setInstantiation(true);
          
          //must be greater than model order
          
          //          int paramPad = 1024;        // The analysis is done on a longer segment than the block, with this many extra values on either side
          //padding min 0
          params.emplace_back("padding", "Padding", parameter::Type::kLong);
          params.back().setInstantiation(false).setMin(0).setDefault(128).setInstantiation(false);
          
          
          // This ensures the analysis is valid across the whole block (some padding is a good idea, but not too much)
          
          // The detection parameters
          
          // Detection is based on absolute forward and backwards prediction errors in relation to the estimated deviation of the AR model - these predictions are smoothed with a window and subjected to an on and off threshold - higher on thresholds make detection less likely and the reset threshold is used (along with a hold time) to ensure that the detection does not switch off before the end of a transient
          
          
          //'skew', do 2^n -10, 10
          
          //          double paramDetectPower = 1.0;           // The power factor used when windowing - higher makes detection more likely
          params.emplace_back("skew", "Skew", parameter::Type::kFloat);
          params.back().setInstantiation(false).setMin(-10).setMax(10).setDefault(0).setInstantiation(false);
          
          
          //          double paramDetectThreshHi = 3.0;        // The threshold for detection (in multiples of the model deviation)
          //
          params.emplace_back("threshfwd", "Forward Threshold",
                              parameter::Type::kFloat);
          params.back().setInstantiation(false).setMin(0).setDefault(3).setInstantiation(false);
          
          
          //          double paramDetectThreshLo = 1.1;        // The reset threshold to end a detected segment (in multiples of the model deviation)
          params.emplace_back("threshback", "Backward Threshold",
                              parameter::Type::kFloat);
          params.back().setInstantiation(false).setMin(0).setDefault(1.1).setInstantiation(false);
          
          
          //          double paramDetectHalfWindow = 7;        // Half the window size used to smooth detection functions (in samples)
          //up to model order ~40 = 1ms, 15 default sampples for whole window
          //
          params.emplace_back("windowsize", "Window Size(ms)",
                              parameter::Type::kFloat);
          params.back().setInstantiation(false).setMin(0).setDefault(14).setInstantiation(false);
          
          
          //          int paramDetectHold = 25;               // The hold time for detection (in samples)
          //prevents onsets within n samples of an offset, min 0,
          params.emplace_back("debounce", "Debounce(ms)",
                              parameter::Type::kFloat);
          params.back().setInstantiation(false).setMin(0).setDefault(25).setInstantiation(false);

        }
        
        return params;
      }
      
      
      
      TransientsSlice() = default;
      TransientsSlice(TransientsSlice&) = delete;
      TransientsSlice operator=(TransientsSlice&) = delete;
      
      TransientsSlice(size_t maxWindowSize):
      //stft::STFTCheckParams(windowsize,hopsize,fftsize),
      audio::BaseAudioClient<T,U>(maxWindowSize,1,1,1)
      {
        newParamSet();
      }
      
      void reset() 
      {
        static constexpr unsigned iterations = 3;
        static constexpr double   robustFactor = 3.0;
        size_t order = parameter::lookupParam("order", mParams).getLong();
        size_t blocksize = parameter::lookupParam("blocksize", mParams).getLong();
        size_t padding = parameter::lookupParam("padding", mParams).getLong();
        mExtractor = std::unique_ptr<TransientSegmentation>(new TransientSegmentation(order, iterations,robustFactor));
        mExtractor->prepareStream(blocksize, padding);
        audio::BaseAudioClient<T,U>::reset();
      }
      
      std::tuple<bool,std::string> sanityCheck()
      {
        const std::vector<parameter::Descriptor>& desc = getParamDescriptors();
        //First, let's make sure that we have a complete of parameters of the right sort
        bool sensible = std::equal(mParams.begin(), mParams.end(),desc.begin(),
         [](const parameter::Instance& i, const parameter::Descriptor& d)
         {
           return i.getDescriptor() == d;
         });
        
        if(! sensible || (desc.size() != mParams.size()))
        {
          return {false, "Invalid params passed. Were these generated with newParameterSet()?" };
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
            case parameter::Instance::RangeErrorType::kMin:
              msg << " value below minimum(" << d.getMin() << ")";
              break;
            case parameter::Instance::RangeErrorType::kMax:
              msg << " value above maximum(" << d.getMin() << ")";
            default:
              assert(false && "This should be unreachable");
            }
            return { false, msg.str()};
          }
        }
        
        size_t halfWindow = std::round(parameter::lookupParam("windowsize", mParams).getFloat() / 2);
        long order = parameter::lookupParam("order", mParams).getLong();
        if(order < halfWindow)
        {
          return {false, "Model order must be more than half the window size"};
        }
        
        long blocksize = parameter::lookupParam("blocksize", mParams).getLong();
        if(blocksize < order)
        {
          return {false, "Block size must be greater than model order"};
        }
        
        return {true,"Groovy"};
      }
      
      //Here we do an STFT and its inverse
      void process(data_type input, data_type output) override
      {        
        double skew = std::pow(2,parameter::lookupParam("skew", getParams()).getFloat());
        double fwdThresh = parameter::lookupParam("threshfwd", getParams()).getFloat();
        double backThresh = parameter::lookupParam("threshback", getParams()).getFloat();
        size_t halfWindow = std::round(parameter::lookupParam("windowsize", getParams()).getFloat() /2);
        size_t debounce = parameter::lookupParam("debounce", getParams()).getLong();
        
        mExtractor->setDetectionParameters(skew, fwdThresh, backThresh, halfWindow, debounce);
        FluidTensorView<const double,1> markers{mExtractor->process(input.data(), mExtractor->inputSize()),0, mExtractor->hopSize()};
        output.row(0)(fluid::Slice(0, mExtractor->hopSize())) = markers;
      }
      //Here we gain compensate for the OLA
      void postProcess(data_type output) override {}

      std::vector<parameter::Instance>& getParams() override
      {
        return mParams;
      }
      
      size_t getWindowSize() override
      {
        assert(mExtractor);
        return mExtractor->inputSize();
      }
      
      size_t getHopSize() override
      {
        assert(mExtractor);
        return mExtractor->hopSize();
      }
      
    private:
      void newParamSet()
      {
        mParams.clear();
        for(auto&& d: getParamDescriptors())
          mParams.emplace_back(d);
      }
      
      std::unique_ptr<TransientSegmentation> mExtractor;
      FluidTensor<std::complex<T>,1> mTransients;
      std::vector<parameter::Instance> mParams;
    };
    
    
  }//namespace hpss
}//namespace fluid

