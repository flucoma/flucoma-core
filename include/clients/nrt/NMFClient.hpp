#pragma once

#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/ParameterConstraints.hpp>
#include <clients/common/OfflineClient.hpp>
#include <clients/common/DeriveSTFTParams.hpp>
#include <clients/common/ParameterSet.hpp>
#include <algorithms/public/NMF.hpp>
#include <algorithms/public/RatioMask.hpp>
#include <algorithms/public/STFT.hpp>
#include <data/FluidTensor.hpp>


#include <algorithm> //for max_element
#include <sstream>   //for ostringstream
#include <string>
#include <unordered_set>
#include <utility> //for std make_pair
#include <vector>  //for containers of params, and for checking things
#include <cassert>

namespace fluid {
namespace client {

enum NMFParamIndex {kSource, kOffset, kNumFrames, kStartChan, kNumChans, kResynth,kFilters,kFiltersUpdate,kEnvelopes,kEnvelopesUpdate,kRank,kIterations,kWinSize,kHopSize,kFFTSize};

auto constexpr NMFParams = defineParameters(
  BufferParam("sourceBuf","Source Buffer"),
  LongParam("offset","Source Offset",0,Min(0)),
  LongParam("nFrames","Number Frames",-1),
  LongParam("startChan","Start Channel",0,Min(0)),
  LongParam("numChans","Number Channels",-1),
  BufferParam("resynthBuf", "Resynthesis Buffer"),
  BufferParam("filterBuf", "Filters Buffer"),
  EnumParam("filterUpdate", "Filters Buffer Update", 0, "None","Seed","Fixed"),
  BufferParam("envBuf", "Envelopes Buffer"),
  EnumParam("envUpdate", "Envelopes Buffer Update", 0, "None","Seed","Fixed"),
  LongParam("rank", "Rank", 5, Min(1)),
  LongParam("iterations", "Iterations", 100, Min(1)),
  LongParam("winSize", "Window Size", 1024, FFTUpperLimit<kFFTSize>()),
  LongParam("hopSize", "Hop Size", 256),
  LongParam("fftSize", "FFT Size", 1024, WinLowerLimit<kWinSize>()));//,PowerOfTwo()));

template<typename Params, typename T, typename U>
class NMFClient: public FluidBaseClient<Params>, public OfflineIn, public OfflineOut
{

private:
  Params& mParams;

public:

  NMFClient(Params& p): FluidBaseClient<Params>(p), mParams(p) {}

  /***
   Take some data, NMF it
   ***/
  Result process() {

//    assert(inputs.size() == 1 );
  
    if(!param<kSource>(*this).get())
    {
      return {Result::Status::kError,"No input"};
    }
  
 
    BufferAdaptor::Access source(param<kSource>(*this).get());

    if(!(source.exists() && source.valid()))
      return {Result::Status::kError, "Source Buffer Not Found or Invalid"};

    size_t winSize, hopSize, fftSize;
    
    std::tie(winSize,hopSize,fftSize) = impl::deriveSTFTParams<kWinSize, kHopSize, kFFTSize>(mParams);

    size_t nChannels = param<kNumChans>(*this) == -1 ? source.numChans() : param<kNumChans>(*this);
    size_t nFrames   = param<kNumFrames>(*this) == -1  ? source.numFrames(): param<kNumFrames>(*this);
    size_t nWindows  = std::floor((nFrames + hopSize) / hopSize);
    size_t nBins     = fftSize / 2 + 1;

    bool hasFilters {false};
    const bool seedFilters {param<kFiltersUpdate>(*this) > 0};
    const bool fixFilters  {param<kFiltersUpdate>(*this) == 2};
    
    if(param<kFilters>(*this))
    {
      BufferAdaptor::Access buf(param<kFilters>(*this).get());
      if(!buf.exists())
        return {Result::Status::kError, "Filter Buffer Supplied But Invalid"};

      if (buf.valid() && (param<kFiltersUpdate>(*this) > 0)
          && (buf.numFrames() != nBins || buf.numChans()  != param<kRank>(*this) * nChannels))
            return { Result::Status::kError, "Supplied filter buffer for seeding must be [(FFTSize / 2) + 1] frames long, and have [rank] * [channels] channels"};
      hasFilters = true;
    }
    else
      if( param<kFiltersUpdate>(*this) > 0)
        return {Result::Status::kError, "Filter Mode set to Seed or Fix , but no Filter Buffer supplied" };

    bool hasEnvelopes {false};
    const bool seedEnvelopes {param<kEnvelopesUpdate>(*this) > 0};
    const bool fixEnvelopes {param<kEnvelopesUpdate>(*this) == 2};

    if(fixEnvelopes && fixFilters)
      return {Result::Status::kError, "It doesn't make any sense to fix both filters and envelopes"};

    if(param<kEnvelopes>(*this))
    {
      BufferAdaptor::Access buf(param<kEnvelopes>(*this).get());
      if(!buf.exists())
        return {Result::Status::kError, "Envelope Buffer Supplied But Invalid"};

      if (buf.valid() && (param<kEnvelopesUpdate>(*this) > 0)
         && (buf.numFrames() != (nFrames / hopSize) + 1 || buf.numChans()  != param<kRank>(*this) * nChannels))
            return {Result::Status::kError, "Supplied envelope buffer for seeding must be [(num samples / hop "
                    "size)  + 1] frames long, and have [rank] * [channels] channels"};

      hasEnvelopes = true;
    }
    else
      if( param<kEnvelopesUpdate>(*this) > 0)
        return {Result::Status::kError, "Envelope Mode set to Seed or Fix , but no Envelope Buffer supplied" };


    bool hasResynth {false};

    if(param<kResynth>(*this))
    {
      BufferAdaptor::Access buf(param<kResynth>(*this).get());
      if(!buf.exists())
        return {Result::Status::kError, "Resynthesis Buffer Supplied But Invalid"};
      hasResynth = true;
    }

    if (hasResynth)
      BufferAdaptor::Access(param<kResynth>(*this).get()).resize(nFrames, nChannels, param<kRank>(*this));
    if (hasFilters && !param<kFiltersUpdate>(*this))
      BufferAdaptor::Access(param<kFilters>(*this).get()).resize(nBins, nChannels, param<kRank>(*this));
    if (hasEnvelopes && !param<kEnvelopesUpdate>(*this))
      BufferAdaptor::Access(param<kEnvelopes>(*this).get()).resize((nFrames / hopSize) + 1, nChannels, param<kRank>(*this));

    auto stft = algorithm::STFT(winSize, fftSize, hopSize);

    auto tmp = FluidTensor<double, 1>(nFrames);
    auto seededFilters = FluidTensor<double, 2>(0, 0);
    auto seededEnvelopes = FluidTensor<double, 2>(0, 0);
    auto outputFilters = FluidTensor<double, 2>(param<kRank>(*this), nBins);
    auto outputEnvelopes = FluidTensor<double, 2>(nWindows, param<kRank>(*this));
    auto spectrum = FluidTensor<std::complex<double>,2>(nWindows,nBins);
    auto magnitude = FluidTensor<double,2>(nWindows,nBins);
    auto outputMags = FluidTensor<double,2>(nWindows,nBins);

    if (seedFilters || fixFilters)
      seededFilters.resize(param<kRank>(*this), nBins);
    if (seedEnvelopes || fixEnvelopes)
      seededEnvelopes.resize((nFrames / hopSize) + 1, param<kRank>(*this));

    for (size_t i = 0; i < nChannels; ++i) {
      //          tmp = sourceData.col(i);
      tmp = source.samps(param<kOffset>(*this), nFrames, param<kStartChan>(*this) + i);
      stft.process(tmp, spectrum);
      algorithm::STFT::magnitude(spectrum,magnitude);

      // For multichannel dictionaries, seed data could be all over the place,
      // so we'll build it up by hand :-/
      for (size_t j = 0; j < param<kRank>(*this); ++j) {
        if (seedFilters || fixFilters)
        {
          auto filters = BufferAdaptor::Access{param<kFilters>(*this).get()};
          seededFilters.row(j) = filters.samps(i, j);
        }
        if (seedEnvelopes || fixEnvelopes)
        {
          auto envelopes = BufferAdaptor::Access(param<kEnvelopes>(*this).get());
          seededEnvelopes.col(j) = envelopes.samps(i, j);
        }
      }

      auto nmf = algorithm::NMF(param<kRank>(*this), param<kIterations>(*this), !fixFilters, !fixEnvelopes);

      nmf.process(magnitude,outputFilters,outputEnvelopes,outputMags,seededFilters, seededEnvelopes);

      // Write W?
      if (hasFilters && !fixFilters) {
//        auto finalFilters = m.getW();
        auto filters = BufferAdaptor::Access{param<kFilters>(*this).get()};
        for (size_t j = 0; j < param<kRank>(*this); ++j)
        {
          filters.samps(i, j) = outputFilters.row(j);
        }
      }

      // Write H? Need to normalise also
      if (hasEnvelopes && !fixEnvelopes) {
//        auto finalEnvelopes = m.getH();
        auto maxH = *std::max_element(outputEnvelopes.begin(), outputEnvelopes.end());
        auto scale = 1. / (maxH);
        auto envelopes = BufferAdaptor::Access{param<kEnvelopes>(*this).get()};

        

        for (size_t j = 0; j < param<kRank>(*this); ++j) {
          auto env = envelopes.samps(i, j);
          env = outputEnvelopes.col(j);
          env.apply([scale](float &x) { x *= scale; });
        }
      }

      if (hasResynth) {
        auto mask = algorithm::RatioMask{outputMags, 1};
        auto resynthMags = FluidTensor<double,2>(nWindows,nBins);
        auto resynthSpectrum = FluidTensor<std::complex<double>,2>(nWindows,nBins);
        auto istft = algorithm::ISTFT{winSize, fftSize, hopSize};
        auto resynthAudio = FluidTensor<double,1>(nFrames);
        auto resynth = BufferAdaptor::Access{param<kResynth>(*this).get()};

        for (size_t j = 0; j < param<kRank>(*this); ++j) {
          algorithm::NMF::estimate(outputFilters,outputEnvelopes,j, resynthMags);
          mask.process(spectrum,resynthMags,resynthSpectrum);
          istft.process(resynthSpectrum,resynthAudio);
          resynth.samps(i, j) = resynthAudio(Slice(0, nFrames));
        }
      }
    }
     return {Result::Status::kOk,""};
  }
};
} // namespace client
} // namespace fluid
