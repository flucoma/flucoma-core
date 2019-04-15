#pragma once

#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/ParameterConstraints.hpp>
#include <clients/common/OfflineClient.hpp>
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

enum NMFParamIndex {kSource, kOffset, kNumFrames, kStartChan, kNumChans, kResynth,kFilters,kFiltersUpdate,kEnvelopes,kEnvelopesUpdate,kRank,kIterations,kFFT};

auto constexpr NMFParams = defineParameters(
  BufferParam("source","Source Buffer"),
  LongParam("startFrame","Source Offset",0, Min(0)),
  LongParam("numFrames","Number of Frames",-1),
  LongParam("startChan","Start Channel",0,Min(0)),
  LongParam("numChans","Number Channels",-1),
  BufferParam("resynth", "Resynthesis Buffer"),
  BufferParam("bases", "Bases Buffer"),
  EnumParam("basesMode", "Bases Buffer Update Mode", 0, "None","Seed","Fixed"),
  BufferParam("activations", "Activations Buffer"),
  EnumParam("actMode", "Activations Buffer Update Mode", 0, "None","Seed","Fixed"),
  LongParam("rank", "Rank", 1, Min(1)),
  LongParam("numIter", "Number of Iterations", 100, Min(1)),
  FFTParam("fftSettings", "FFT Settings", 1024, -1, -1)
);

template<typename T>
class NMFClient: public FluidBaseClient<decltype(NMFParams), NMFParams>, public OfflineIn, public OfflineOut
{

public:

  NMFClient(ParamSetViewType& p): FluidBaseClient(p)
  {}

  /***
   Take some data, NMF it
   ***/
  Result process() {

//    assert(inputs.size() == 1 );

    if(!get<kSource>().get())
    {
      return {Result::Status::kError,"No input"};
    }

    BufferAdaptor::Access source(get<kSource>().get());

    if(!(source.exists() && source.valid()))
      return {Result::Status::kError, "Source Buffer Not Found or Invalid"};

    double sampleRate = source.sampleRate();
    auto fftParams = get<kFFT>();

    size_t nChannels = get<kNumChans>() == -1 ? source.numChans() : get<kNumChans>();
    size_t nFrames   = get<kNumFrames>() == -1  ? source.numFrames(): get<kNumFrames>();
    size_t nWindows  = std::floor((nFrames + fftParams.hopSize()) / fftParams.hopSize());
    size_t nBins     = fftParams.frameSize();

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
         && (buf.numFrames() != (nFrames / fftParams.hopSize()) + 1 || buf.numChans()  != get<kRank>() * nChannels))
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
      BufferAdaptor::Access(get<kResynth>().get()).resize(nFrames, nChannels, get<kRank>(),sampleRate);
    if (hasFilters && !get<kFiltersUpdate>())
      BufferAdaptor::Access(get<kFilters>().get()).resize(nBins, nChannels, get<kRank>(),0);
    if (hasEnvelopes && !get<kEnvelopesUpdate>())
      BufferAdaptor::Access(get<kEnvelopes>().get()).resize((nFrames / fftParams.hopSize()) + 1, nChannels, get<kRank>(),0);

    auto stft = algorithm::STFT(fftParams.winSize(), fftParams.fftSize(), fftParams.hopSize());

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
      seededEnvelopes.resize((nFrames / fftParams.hopSize()) + 1, get<kRank>());

    for (size_t i = 0; i < nChannels; ++i) {
      //          tmp = sourceData.col(i);
      tmp = source.samps(get<kOffset>(), nFrames, get<kStartChan>() + i);
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
        auto istft = algorithm::ISTFT{fftParams.winSize(), fftParams.fftSize(), fftParams.hopSize()};
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
