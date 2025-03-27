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

#include "../common/BufferedProcess.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/OfflineClient.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../common/Result.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"

namespace fluid {
namespace client {
namespace bufcompose {

enum {
  kSource,
  kOffset,
  kNumFrames,
  kStartChan,
  kNChans,
  kGain,
  kDest,
  kDestOffset,
  kDestStartChan,
  kDestGain
};

constexpr auto BufComposeParams = defineParameters(
    InputBufferParam("source", "Source Buffer"),
    LongParam("startFrame", "Source Offset", 0, Min(0)),
    LongParam("numFrames", "Source Number of Frames", -1),
    LongParam("startChan", "Source Channel Offset", 0, Min(0)),
    LongParam("numChans", "Source Number of Channels", -1),
    FloatParam("gain", "Source Gain", 1.0),
    BufferParam("destination", "Destination Buffer"),
    LongParam("destStartFrame", "Destination Offset", 0),
    LongParam("destStartChan", "Destination Channel Offset", 0),
    FloatParam("destGain", "Destination Gain", 0.0));

class BufComposeClient : public FluidBaseClient, OfflineIn, OfflineOut
{

public:
  using ParamDescType = decltype(BufComposeParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return BufComposeParams; }


  BufComposeClient(ParamSetViewType& p, FluidContext&) : mParams{p} {}

  template <typename T>
  Result process(FluidContext& c)
  {
    // Not using bufferRangeCheck to validate source ranges because BufCompose
    // is special...
    if (!get<kSource>().get()) { return {Result::Status::kError, "No input"}; }

    index nChannels{0};
    index nFrames{0};

    {
      BufferAdaptor::ReadAccess source(get<kSource>().get());

      if (!(source.exists() && source.valid()))
        return {Result::Status::kError, "Source Buffer Not Found or Invalid"};

      nChannels = get<kNChans>() < 0 ? source.numChans() - get<kStartChan>()
                                     : get<kNChans>();
      nFrames = get<kNumFrames>() < 0 ? source.numFrames() - get<kOffset>()
                                      : get<kNumFrames>();

      if (nChannels <= 0 || nFrames <= 0)
        return {Result::Status::kError, "Zero length segment requested"};

      // We don't care if the overall number of frames will overrun, because
      // we'll zero pad, but the offset should be within the source range
      if (get<kOffset>() >= source.numFrames())
        return {Result::Status::kError, "Start frame (", get<kOffset>(),
                ") out of range."};

      // We don't care if the overall number of channels will overrun, because
      // we'll loop, but the offset should be within the source range
      if (get<kStartChan>() >= source.numChans())
        return {Result::Status::kError, "Start channel ", get<kStartChan>(),
                " out of range."};
    }

    index             dstStart = get<kDestOffset>();
    index             dstStartChan = get<kDestStartChan>();
    index             dstEnd{0};
    index             dstEndChan{0};
    bool              destinationResizeNeeded{false};
    FluidTensor<T, 2> destinationOrig(0, 0);

    {
      BufferAdaptor::Access destination(get<kDest>().get());

      if (!destination.exists())
        return {Result::Status::kError,
                "Destination Buffer Not Found or Invalid"};

      dstEnd = dstStart + nFrames;
      dstEndChan = dstStartChan + nChannels;

      destinationResizeNeeded = (dstEnd > destination.numFrames()) ||
                                (dstEndChan > destination.numChans());

      auto applyGain = [this](T& x) { x *= static_cast<T>(get<kDestGain>()); };


      if (destinationResizeNeeded) // copy the whole of desintation if we have
                                   // to resize it
      {
        destinationOrig.resize(
            std::max<index>(dstEndChan, destination.numChans()),
            std::max<index>(dstEnd, destination.numFrames()));
        if (destination.numChans() > 0 && destination.numFrames() > 0)
        {
          for (index i = 0; i < destination.numChans(); ++i)
            destinationOrig.row(i)(Slice(0, destination.numFrames())) <<=
                destination.samps(i);
          destinationOrig(Slice(dstStartChan, dstEndChan - dstStartChan),
                          Slice(dstStart, dstEnd - dstStart))
              .apply(applyGain);
        }
      }
      else // just copy what we're affecting
      {
        destinationOrig.resize(nChannels, nFrames);
        for (index i = 0; i < nChannels; ++i)
        {
          destinationOrig.row(i) <<=
              destination.samps(dstStart, nFrames, dstStartChan + i);
          destinationOrig.row(i).apply(applyGain);
        }
      }
    }

    // Access sources inside own scope block, so they'll be unlocked before
    // we need to (possibly) resize the desintation buffer which could
    // (possibly)  also be one of the sources
    {
      BufferAdaptor::ReadAccess source(get<kSource>().get());
      auto                      gain = get<kGain>();
      // iterates through the copying of the first source
      for (index i = dstStartChan, j = 0; j < nChannels; ++i, ++j)
      {
        // Special repeating channel voodoo
        HostVector<const T> sourceChunk{source.samps(
            get<kOffset>(), std::min<index>(nFrames, source.numFrames()),
            (get<kStartChan>() + j) % source.numChans())};

        HostVector<T> destinationChunk{destinationOrig.row(j)};
        if (destinationResizeNeeded)
        {
          destinationChunk.reset(destinationOrig.row(i).data(), dstStart,
                                 nFrames);
        }

        std::transform(sourceChunk.begin(), sourceChunk.end(),
                       destinationChunk.begin(), destinationChunk.begin(),
                       [gain](const T& src, T& dst) {
                         return static_cast<T>(dst + src * gain);
                       });

        if (c.task() &&
            !c.task()->processUpdate(static_cast<double>(j + 1),
                                     static_cast<double>(nChannels)))
          return {Result::Status::kCancelled, ""};
      }
    }

    BufferAdaptor::Access destination(get<kDest>().get());

    if (destinationResizeNeeded)
    {
      Result resizeResult =
          destination.resize(destinationOrig.cols(), destinationOrig.rows(),
                             destination.sampleRate());
      if (!resizeResult.ok()) return resizeResult;
      for (index i = 0; i < destination.numChans(); ++i)
        destination.samps(i) <<= destinationOrig.row(i);
    }
    else
    {
      for (index i = 0; i < nChannels; ++i)
        destination.samps(dstStart, nFrames, dstStartChan + i) <<=
            destinationOrig.row(i);
      destination.refresh(); // make sure the buffer is marked dirty
    }

    return {Result::Status::kOk};
  }
};
} // namespace bufcompose
using NRTThreadedBufComposeClient =
    NRTThreadingAdaptor<ClientWrapper<bufcompose::BufComposeClient>>;

} // namespace client
} // namespace fluid
