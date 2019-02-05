#pragma once

#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/ParameterConstraints.hpp>
#include <clients/common/OfflineClient.hpp>
#include <clients/common/DeriveSTFTParams.hpp>
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

enum NMFParamIndex {kResynth,kFilters,kFiltersUpdate,kEnvelopes,kEnvelopesUpdate,kRank,kIterations,kWinSize,kHopSize,kFFTSize};


auto constexpr NMFParams = std::make_tuple(
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

using NMFParamsT = decltype(NMFParams);

/**
 Integration class for doing NMF filtering and resynthesis
 **/

class NMFClient: public FluidBaseClient<NMFParamsT>, public OfflineIn, public OfflineOut
{
public:

  NMFClient(): FluidBaseClient<NMFParamsT>(NMFParams)
  {
    audioBuffersIn(1);
    audioBuffersOut(0);
  }

  /***
   Take some data, NMF it
   ***/
  Result process(std::vector<BufferProcessSpec>& inputs, std::vector<BufferProcessSpec>&) {

    assert(inputs.size() == 1 );
 
    BufferAdaptor::Access source(inputs[0].buffer);

    if(!(source.exists() && source.valid()))
      return {Result::Status::kError, "Source Buffer Not Found or Invalid"};

    size_t winSize, hopSize, fftSize;
    
    std::tie(winSize,hopSize,fftSize) = impl::deriveSTFTParams<kWinSize, kHopSize, kFFTSize>(*this);

    size_t nChannels = inputs[0].nChans == -1 ? source.numChans() : inputs[0].nChans;
    size_t nFrames   = inputs[0].nFrames == -1  ? source.numFrames(): inputs[0].nFrames;
    size_t nWindows  = std::floor((nFrames + hopSize) / hopSize);
    size_t nBins     = fftSize / 2 + 1;

    bool hasFilters {false};
    const bool seedFilters {get<kFiltersUpdate>() > 0};
    const bool fixFilters  {get<kFiltersUpdate>() == 2};
    
    if(get<kFilters>())
    {
      BufferAdaptor::Access buf(get<kFilters>().get());
      if(!buf.exists())
        return {Result::Status::kError, "Filter Buffer Supplied But Invalid"};

      if (buf.valid() && (get<kFiltersUpdate>() > 0)
          && (buf.numFrames() != nBins || buf.numChans()  != get<kRank>() * nChannels))
            return { Result::Status::kError, "Supplied filter buffer for seeding must be [(FFTSize / 2) + 1] frames long, and have [rank] * [channels] channels"};
      hasFilters = true;
    }
    else
      if( get<kFiltersUpdate>() > 0)
        return {Result::Status::kError, "Filter Mode set to Seed or Fix , but no Filter Buffer supplied" };

    bool hasEnvelopes {false};
    const bool seedEnvelopes {get<kEnvelopesUpdate>() > 0};
    const bool fixEnvelopes {get<kEnvelopesUpdate>() == 2};

    if(fixEnvelopes && fixFilters)
      return {Result::Status::kError, "It doesn't make any sense to fix both filters and envelopes"};

    if(get<kEnvelopes>())
    {
      BufferAdaptor::Access buf(get<kEnvelopes>().get());
      if(!buf.exists())
        return {Result::Status::kError, "Envelope Buffer Supplied But Invalid"};

      if (buf.valid() && (get<kEnvelopesUpdate>() > 0)
         && (buf.numFrames() != (nFrames / hopSize) + 1 || buf.numChans()  != get<kRank>() * nChannels))
            return {Result::Status::kError, "Supplied envelope buffer for seeding must be [(num samples / hop "
                    "size)  + 1] frames long, and have [rank] * [channels] channels"};

      hasEnvelopes = true;
    }
    else
      if( get<kEnvelopesUpdate>() > 0)
        return {Result::Status::kError, "Envelope Mode set to Seed or Fix , but no Envelope Buffer supplied" };


    bool hasResynth {false};

    if(get<kResynth>())
    {
      BufferAdaptor::Access buf(get<kResynth>().get());
      if(!buf.exists())
        return {Result::Status::kError, "Resynthesis Buffer Supplied But Invalid"};
      hasResynth = true;
    }

    if (hasResynth)
      BufferAdaptor::Access(get<kResynth>().get()).resize(nFrames, nChannels, get<kRank>());
    if (hasFilters && !get<kFiltersUpdate>())
      BufferAdaptor::Access(get<kFilters>().get()).resize(nBins, nChannels, get<kRank>());
    if (hasEnvelopes && !get<kEnvelopesUpdate>())
      BufferAdaptor::Access(get<kEnvelopes>().get()).resize((nFrames / hopSize) + 1, nChannels, get<kRank>());

    auto stft = algorithm::STFT(winSize, fftSize, hopSize);

    auto tmp = FluidTensor<double, 1>(nFrames);
    auto seededFilters = FluidTensor<double, 2>(0, 0);
    auto seededEnvelopes = FluidTensor<double, 2>(0, 0);
    auto outputFilters = FluidTensor<double, 2>(get<kRank>(), nBins);
    auto outputEnvelopes = FluidTensor<double, 2>(nWindows, get<kRank>());
    auto spectrum = FluidTensor<std::complex<double>,2>(nWindows,nBins);
    auto magnitude = FluidTensor<double,2>(nWindows,nBins);
    auto outputMags = FluidTensor<double,2>(nWindows,nBins);

    if (seedFilters || fixFilters)
      seededFilters.resize(get<kRank>(), nBins);
    if (seedEnvelopes || fixEnvelopes)
      seededEnvelopes.resize((nFrames / hopSize) + 1, get<kRank>());

    for (size_t i = 0; i < nChannels; ++i) {
      //          tmp = sourceData.col(i);
      tmp = source.samps(inputs[0].startFrame, nFrames, inputs[0].startChan + i);
      stft.process(tmp, spectrum);
      algorithm::STFT::magnitude(spectrum,magnitude);

      // For multichannel dictionaries, seed data could be all over the place,
      // so we'll build it up by hand :-/
      for (size_t j = 0; j < get<kRank>(); ++j) {
        if (seedFilters || fixFilters)
        {
          auto filters = BufferAdaptor::Access{get<kFilters>().get()};
          seededFilters.row(j) = filters.samps(i, j);
        }
        if (seedEnvelopes || fixEnvelopes)
        {
          auto envelopes = BufferAdaptor::Access(get<kEnvelopes>().get());
          seededEnvelopes.col(j) = envelopes.samps(i, j);
        }
      }

      auto nmf = algorithm::NMF(get<kRank>(), get<kIterations>(), !fixFilters, !fixEnvelopes);

      nmf.process(magnitude,outputFilters,outputEnvelopes,outputMags,seededFilters, seededEnvelopes);

      // Write W?
      if (hasFilters && !fixFilters) {
//        auto finalFilters = m.getW();
        auto filters = BufferAdaptor::Access{get<kFilters>().get()};
        for (size_t j = 0; j < get<kRank>(); ++j)
        {
          filters.samps(i, j) = outputFilters.row(j);
        }
      }

      // Write H? Need to normalise also
      if (hasEnvelopes && !fixEnvelopes) {
//        auto finalEnvelopes = m.getH();
        auto maxH = *std::max_element(outputEnvelopes.begin(), outputEnvelopes.end());
        auto scale = 1. / (maxH);
        auto envelopes = BufferAdaptor::Access{get<kEnvelopes>().get()};

        for (size_t j = 0; j < get<kRank>(); ++j) {
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
        auto resynth = BufferAdaptor::Access{get<kResynth>().get()};

        for (size_t j = 0; j < get<kRank>(); ++j) {
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
