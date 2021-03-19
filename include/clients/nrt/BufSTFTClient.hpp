/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
#pragma once

#include "../common/BufferedProcess.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/STFT.hpp"

namespace fluid {
namespace client {

class BufferSTFTClient : public FluidBaseClient,
                          public OfflineIn,
                          public OfflineOut
{

  enum BufferSTFTParamIndex {
    kSource,
    kOffset,
    kNumFrames,
    kStartChan,
    // kNumChans,
    kMag,
    kPhase,
    kResynth, 
    kInvert, 
    kFFT
  };

public:
  FLUID_DECLARE_PARAMS(InputBufferParam("source", "Source Buffer"),
                       LongParam("startFrame", "Source Offset", 0, Min(0)),
                       LongParam("numFrames", "Number of Frames", -1),
                       LongParam("startChan", "Start Channel", 0, Min(0)),
                       // LongParam("numChans", "Number of Channels", -1),
                       BufferParam("magnitude", "Magnitude Buffer"),
                       BufferParam("phase", "Phase Buffer"),
                       BufferParam("resynthesis", "Resynthesis Buffer"),
                       LongParam("inverse","Inverse transform",0,Min(0),Max(1)),
                       FFTParam("fftSettings", "FFT Settings", 1024, -1, -1)
                     );

  BufferSTFTClient(ParamSetViewType& p) : 
        mParams(p), mSTFTBufferedProcess{65536, 1, 1}
  {}

  template <typename T>
  Result process(FluidContext& c)
  {    
    if(get<kInvert>() == 0)
      return processFwd<T>(c);
    else 
      return processInverse<T>(c);
  }
    
private: 
  template <typename T>
  Result processFwd(FluidContext& c)
  { 
    auto s = get<kSource>().get();
    if (!s)
      return {Result::Status::kError, "No input buffer supplied"};

    auto m = get<kMag>().get(); 
    auto p = get<kPhase>().get();    

    bool haveMag = m != nullptr; 
    bool havePhase = p != nullptr; 

    if (!haveMag && !havePhase)
      return {Result::Status::kError, "Neither magnitude nor phase buffer supplied"};

    auto mags = BufferAdaptor::Access(m);
    auto phases = BufferAdaptor::Access(p);
    
    index numFrames = get<kNumFrames>(); 
    index numChans  = 1; //get<kNumChans>(); 
    Result rangeOK = bufferRangeCheck(get<kSource>().get(), get<kOffset>(),
                      numFrames,get<kStartChan>(),numChans); 
        
    if(!rangeOK.ok()) return rangeOK;
     
    auto source = BufferAdaptor::ReadAccess(s); 
         
    if(haveMag && !mags.exists())
        return {Result::Status::kError, "Magnitude buffer not found"};
    
    if(havePhase && !phases.exists())
      return  {Result::Status::kError, "Phase buffer not found"};
    
    index padding = get<kFFT>().winSize(); 
    index numHops = (numFrames + padding) / get<kFFT>().hopSize();
    index numBins = (get<kFFT>().fftSize() / 2) + 1;
    
    //I'm thinking that most things can only really deal with 65536 channels (if that) => limitation on multichannel inputs vs FFT size
    if(numChans * numBins >= 65536) return {Result::Status::kError, "Can produce up to 65536 channels. Split your data up and try again"}; 
      
    if(m)
    {
      auto r = mags.resize(numHops,numBins * numChans,44100);
      if(!r.ok()) return r;   
    }
    
    if(p)
    {
      auto r = phases.resize(numHops,numBins * numChans,44100);
      if(!r.ok()) return r; 
    } 

    FluidTensor<double, 2> tmpMags(1,numBins);
    FluidTensor<double, 2> tmpPhase(1,numBins);
    
    auto magsView = haveMag ? mags.allFrames() : FluidTensorView<T,2>{nullptr, 0, 0, 0};
    auto phaseView = havePhase ? phases.allFrames() : FluidTensorView<T,2>{nullptr, 0, 0, 0};;
    
    auto inputWrapped = std::vector<FluidTensorView<const T,1>>{source.samps(0)};
        
    mSTFTBufferedProcess.processInput( mParams, inputWrapped, c,
      [&tmpMags, &tmpPhase,haveMag,havePhase,&magsView, &phaseView, n=0](ComplexMatrixView in) mutable
      {
        if(haveMag)
        {
           algorithm::STFT::magnitude(in,tmpMags);
           auto sli =   magsView(Slice(0),n);
           sli.col(0) = tmpMags.row(0);
        }
        if(havePhase)
        {
           algorithm::STFT::phase(in,tmpPhase);
           phaseView(Slice(0),n).col(0) = tmpPhase.row(0);
        }
        n++;
      }
    );
    return {};
  }
  
  template <typename T>
  Result processInverse(FluidContext& c )
  {
    auto m = get<kMag>().get(); 
    auto p = get<kPhase>().get();    

    bool haveMag = m != nullptr; 
    bool havePhase = p != nullptr; 
    
    if(!haveMag || !havePhase)
      return {Result::Status::kError, "Need both magnutude and phase buffers for inverse transform"}; 
    
    auto r = get<kResynth>().get();
    
    if(!r)
      return {Result::Status::kError, "No resynthesis buffer supplied"}; 
      
    auto mags = BufferAdaptor::ReadAccess(m);   
    auto phases = BufferAdaptor::ReadAccess(p); 
    
    if(mags.numFrames() != phases.numFrames() || mags.numChans() != phases.numChans())
      return {Result::Status::kError, "Magnitdue and Phase buffer sizes don't match"}; 
    
    index fftSize = get<kFFT>().fftSize();
    index winSize = get<kFFT>().winSize();
    index hopSize = get<kFFT>().hopSize();
    
    if(mags.numChans() != (fftSize / 2) + 1) 
      return {Result::Status::kError, "Wrong number of channels for FFT sizee of ", fftSize, " got ",mags.numChans(), " expected ",  (fftSize / 2) + 1}; 
    
    auto istft = algorithm::ISTFT(winSize, fftSize, hopSize);
    
    index outputSize =  (hopSize * mags.numFrames()) + winSize;//- halfWin;
  
    auto resynth = BufferAdaptor::Access(r);
    
    auto resizeResult = resynth.resize(outputSize - winSize,1,44100);
    if(! resizeResult.ok())
      return resizeResult;
          
    FluidTensor<std::complex<double>,2>  tmpComplex(mags.numFrames() + (winSize/hopSize),mags.numChans());
    FluidTensor<T,1>     tmpOut(outputSize);
    auto magsView = mags.allFrames().transpose();
    auto phaseView = phases.allFrames().transpose();
    
    std::transform(magsView.begin(),magsView.end(),phaseView.begin(),tmpComplex.begin(),
    [](auto& m, auto& p){
      return std::polar(m,p);
    }); 
    
    auto outputWrapped = std::vector<FluidTensorView<T,1>>{tmpOut};
        
    mSTFTBufferedProcess.reset();
    
    mSTFTBufferedProcess.processOutput(mParams,outputWrapped,c,
    [&tmpComplex,n=0](ComplexMatrixView out) mutable
    {
      out.row(0) = tmpComplex.row(n++);
    });
    
    resynth.samps(0) = tmpOut(Slice(winSize));
    return {}; 
  }
    
  STFTBufferedProcess<ParamSetViewType, kFFT, true> mSTFTBufferedProcess;
};

using NRTThreadedBufferSTFTClient =
    NRTThreadingAdaptor<ClientWrapper<BufferSTFTClient>>;

} // namespace client
} // namespace fluid
