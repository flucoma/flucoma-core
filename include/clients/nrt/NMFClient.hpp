/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
#pragma once

#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/OfflineClient.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/NMF.hpp"
#include "../../algorithms/public/RatioMask.hpp"
#include "../../algorithms/public/STFT.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/FluidMemory.hpp"
#include <algorithm> //for max_element
#include <cassert>
#include <sstream> //for ostringstream
#include <string>
#include <unordered_set>
#include <utility> //for std make_pair
#include <vector>  //for containers of params, and for checking things


namespace fluid {
namespace client {
namespace bufnmf {

enum NMFParamIndex {
  kSource,
  kOffset,
  kNumFrames,
  kStartChan,
  kNumChans,
  kResynth,
  kResynthMode,
  kFilters,
  kFiltersUpdate,
  kEnvelopes,
  kEnvelopesUpdate,
  kRank,
  kIterations,
  kFFT
};

constexpr auto BufNMFParams = defineParameters(
    InputBufferParam("source", "Source Buffer"),
    LongParam("startFrame", "Source Offset", 0, Min(0)),
    LongParam("numFrames", "Number of Frames", -1),
    LongParam("startChan", "Start Channel", 0, Min(0)),
    LongParam("numChans", "Number Channels", -1),
    BufferParam("resynth", "Resynthesis Buffer"),
    LongParam("resynthMode","Resynthesise components", 0,Min(0),Max(1)),
    BufferParam("bases", "Bases Buffer"),
    EnumParam("basesMode", "Bases Buffer Update Mode", 0, "None", "Seed",
              "Fixed"),
    BufferParam("activations", "Activations Buffer"),
    EnumParam("actMode", "Activations Buffer Update Mode", 0, "None", "Seed",
              "Fixed"),
    LongParam("components", "Number of Components", 1, Min(1)),
    LongParam("iterations", "Number of Iterations", 100, Min(1)),
    FFTParam("fftSettings", "FFT Settings", 1024, -1, -1));

class NMFClient : public FluidBaseClient, public OfflineIn, public OfflineOut
{
public:
  using ParamDescType = decltype(BufNMFParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return BufNMFParams; }

  NMFClient(ParamSetViewType& p, FluidContext&) : mParams{p} {}

  /***
   Take some data, NMF it
   ***/
  template <typename T>
  Result process(FluidContext& c)
  {

    index nFrames = get<kNumFrames>();
    index nChannels = get<kNumChans>();
    auto  rangeCheck = bufferRangeCheck(get<kSource>().get(), get<kOffset>(),
                                       nFrames, get<kStartChan>(), nChannels);

    if (!rangeCheck.ok()) return rangeCheck;

    auto   source = BufferAdaptor::ReadAccess(get<kSource>().get());
    double sampleRate = source.sampleRate();
    auto   fftParams = get<kFFT>();

    index nWindows = static_cast<index>(
        std::floor((nFrames + fftParams.hopSize()) / fftParams.hopSize()));
    index nBins = fftParams.frameSize();

    bool       hasFilters{false};
    const bool seedFilters{get<kFiltersUpdate>() > 0};
    const bool fixFilters{get<kFiltersUpdate>() == 2};
    const bool shouldResynth = get<kResynthMode>();

    if (get<kFilters>())
    {
      BufferAdaptor::Access buf(get<kFilters>().get());
      if (!buf.exists())
        return {Result::Status::kError, "Bases Buffer Supplied But Invalid"};

      if ((get<kFiltersUpdate>() > 0) &&
          (!buf.valid() || buf.numFrames() != nBins ||
           buf.numChans() != get<kRank>() * nChannels))
        return {Result::Status::kError,
                "Supplied bases buffer for seeding must be [(FFTSize / 2) + "
                "1] frames long, and have [rank] * [channels] channels"};
      hasFilters = true;
    }
    else if (get<kFiltersUpdate>() > 0)
      return {Result::Status::kError,
              "Bases Mode set to Seed or Fix , but no Bases Buffer supplied"};

    bool       hasEnvelopes{false};
    const bool seedEnvelopes{get<kEnvelopesUpdate>() > 0};
    const bool fixEnvelopes{get<kEnvelopesUpdate>() == 2};
    const bool needsAnalysis = !(fixEnvelopes && fixFilters);

    if (!needsAnalysis && !shouldResynth)
      return {Result::Status::kWarning,
              "Bases and Activations buffers both fixed, but resynthesis disabled: no work to do"};

    if (get<kEnvelopes>())
    {
      BufferAdaptor::Access buf(get<kEnvelopes>().get());
      if (!buf.exists())
        return {Result::Status::kError, "Activations Buffer Supplied But Invalid"};

      if ((get<kEnvelopesUpdate>() > 0) &&
          (!buf.valid() ||
           buf.numFrames() != (nFrames / fftParams.hopSize()) + 1 ||
           buf.numChans() != get<kRank>() * nChannels))
        return {
            Result::Status::kError,
            "Supplied activations buffer for seeding must be [(num samples / hop "
            "size)  + 1] frames long, and have [rank] * [channels] channels"};

      hasEnvelopes = true;
    }
    else if (get<kEnvelopesUpdate>() > 0)
      return {
          Result::Status::kError,
          "Activations Mode set to Seed or Fix , but no Activations Buffer supplied"};

    bool hasResynth{false};
    
    
    if(shouldResynth)
    {
      if (get<kResynth>())
      {
        BufferAdaptor::Access buf(get<kResynth>().get());
        if (!buf.exists())
          return {Result::Status::kError,
            "Resynthesis Buffer Supplied But Invalid"};
        hasResynth = true;
      }
      else
      {
        return {Result::Status::kError,
            "Resynthesis requested but no buffer supplied"};
      }
      
      if (hasResynth)
      {
        Result resizeResult =
        BufferAdaptor::Access(get<kResynth>().get())
        .resize(nFrames, nChannels * get<kRank>(), sampleRate);
        if (!resizeResult.ok()) return resizeResult;
      }
    }
    
    if (hasFilters && !get<kFiltersUpdate>())
    {
      Result resizeResult = BufferAdaptor::Access(get<kFilters>().get())
                                .resize(nBins, nChannels * get<kRank>(),
                                        sampleRate / fftParams.fftSize());
      if (!resizeResult.ok()) return resizeResult;
    }
    if (hasEnvelopes && !get<kEnvelopesUpdate>())
    {
      Result resizeResult = BufferAdaptor::Access(get<kEnvelopes>().get())
                                .resize((nFrames / fftParams.hopSize()) + 1,
                                        nChannels * get<kRank>(),
                                        sampleRate / fftParams.hopSize());
      if (!resizeResult.ok()) return resizeResult;
    }

    auto stft = algorithm::STFT(fftParams.winSize(), fftParams.fftSize(),
                                fftParams.hopSize());

    auto tmp = FluidTensor<double, 1>(nFrames);
    auto seededFilters = FluidTensor<double, 2>(0, 0);
    auto seededEnvelopes = FluidTensor<double, 2>(0, 0);
    auto outputFilters = FluidTensor<double, 2>(get<kRank>(), nBins);
    auto outputEnvelopes = FluidTensor<double, 2>(nWindows, get<kRank>());
    auto spectrum = FluidTensor<std::complex<double>, 2>(nWindows, nBins);
    auto magnitude = FluidTensor<double, 2>(nWindows, nBins);
    auto outputMags = FluidTensor<double, 2>(nWindows, nBins);

    if (seedFilters || fixFilters) seededFilters.resize(get<kRank>(), nBins);
    if (seedEnvelopes || fixEnvelopes)
      seededEnvelopes.resize((nFrames / fftParams.hopSize()) + 1, get<kRank>());


    const double progressTotal = static_cast<double>(
        (needsAnalysis * get<kIterations>()) + ((shouldResynth && hasResynth) ? 3 * get<kRank>() : 0));

    for (index i = 0; i < nChannels; ++i)
    {
      if (c.task() &&
          !c.task()->iterationUpdate(static_cast<double>(i),
                                     static_cast<double>(nChannels)))
        return {Result::Status::kCancelled, ""};
      //          tmp = sourceData.col(i);
      tmp <<= source.samps(get<kOffset>(), nFrames, get<kStartChan>() + i);
      stft.process(tmp, spectrum);
      algorithm::STFT::magnitude(spectrum, magnitude);
      int progressCount{0};
      // For multichannel dictionaries, seed data could be all over the place,
      // so we'll build it up by hand :-/
      for (index j = 0; j < get<kRank>(); ++j)
      {
        if (seedFilters || fixFilters)
        {
          auto filters = BufferAdaptor::Access{get<kFilters>().get()};
          seededFilters.row(j) <<= filters.samps(i * get<kRank>() + j);
        }
        if (seedEnvelopes || fixEnvelopes)
        {
          auto envelopes = BufferAdaptor::Access(get<kEnvelopes>().get());
          seededEnvelopes.col(j) <<= envelopes.samps(i * get<kRank>() + j);
        }
      }
      
      auto nmf = algorithm::NMF();
      nmf.addProgressCallback(
          [&c, &progressCount, progressTotal](const index) -> bool {
            return c.task() ? c.task()->processUpdate(
                                  static_cast<double>(++progressCount),
                                  static_cast<double>(progressTotal))
                            : true;
          });
      nmf.process(magnitude, outputFilters, outputEnvelopes, outputMags,
                  get<kRank>(), get<kIterations>() * needsAnalysis, !fixFilters, !fixEnvelopes,
                  seededFilters, seededEnvelopes);

      if (c.task() && c.task()->cancelled())
        return {Result::Status::kCancelled, ""};

      // Write W?
      if (hasFilters && !fixFilters)
      {
        //        auto finalFilters = m.getW();
        auto filters = BufferAdaptor::Access{get<kFilters>().get()};
        for (index j = 0; j < get<kRank>(); ++j)
        { filters.samps(i * get<kRank>() + j) <<= outputFilters.row(j); }
      }

      // Write H? Need to normalise also
      if (hasEnvelopes && !fixEnvelopes)
      {
        //        auto finalEnvelopes = m.getH();
        auto maxH =
            *std::max_element(outputEnvelopes.begin(), outputEnvelopes.end());
        auto scale = 1. / (maxH);
        auto envelopes = BufferAdaptor::Access{get<kEnvelopes>().get()};

        for (index j = 0; j < get<kRank>(); ++j)
        {
          auto env = envelopes.samps(i * get<kRank>() + j);
          env <<= outputEnvelopes.col(j);
          env.apply([scale](float& x) { x *= static_cast<float>(scale); });
        }
      }

      if (shouldResynth && hasResynth)
      {
        auto mask =
            algorithm::RatioMask(nWindows, nBins, FluidDefaultAllocator());
        mask.init(outputMags);
        auto resynthMags = FluidTensor<double, 2>(nWindows, nBins);
        auto resynthSpectrum =
            FluidTensor<std::complex<double>, 2>(nWindows, nBins);
        auto istft = algorithm::ISTFT{fftParams.winSize(), fftParams.fftSize(),
                                      fftParams.hopSize()};
        auto resynthAudio = FluidTensor<double, 1>(nFrames);
        auto resynth = BufferAdaptor::Access{get<kResynth>().get()};

        // const index subProgress = 3 * get<kRank>();

        for (index j = 0; j < get<kRank>(); ++j)
        {
          algorithm::NMF::estimate(outputFilters, outputEnvelopes, j,
                                   resynthMags);
          if (c.task() &&
              !c.task()->processUpdate(++progressCount, progressTotal))
            return {Result::Status::kCancelled, ""};
          mask.process(spectrum, resynthMags, 1, resynthSpectrum);
          if (c.task() &&
              !c.task()->processUpdate(++progressCount, progressTotal))
            return {Result::Status::kCancelled, ""};
          istft.process(resynthSpectrum, resynthAudio);
          resynth.samps(i * get<kRank>() + j) <<= resynthAudio(Slice(0, nFrames));
          if (c.task() &&
              !c.task()->processUpdate(++progressCount, progressTotal))
            return {Result::Status::kCancelled, ""};
        }
      }
    }
    return {Result::Status::kOk, ""};
  }
};
} // namespace bufnmf

using NRTThreadedNMFClient =
    NRTThreadingAdaptor<ClientWrapper<bufnmf::NMFClient>>;


} // namespace client
} // namespace fluid
