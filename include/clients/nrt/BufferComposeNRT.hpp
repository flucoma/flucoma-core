#pragma once

#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/OfflineClient.hpp>
#include <clients/common/ParameterSet.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/Result.hpp>
#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>

namespace fluid {
namespace client {

enum { kSource, kOffset, kNumFrames, kStartChan, kNChans, kGain, kDest, kDestOffset, kDestStartChan, kDestGain };

auto constexpr BufComposeParams = defineParameters(
    BufferParam("source", "Source Buffer"),
    LongParam("startFrame", "Source Offset", 0, Min(0)),
    LongParam("numFrames", "Source Number of Frames", -1),
    LongParam("startChan", "Source Channel Offset", 0, Min(0)),
    LongParam("numChans", "Source Number of Channels", -1),
    FloatParam("gain", "Source Gain", 1.0),
    BufferParam("destination", "Destination Buffer"),
    LongParam("destStartFrame", "Destination Offset", 0),
    LongParam("destStartChan", "Destination Channel Offset", 0),
    FloatParam("destGain", "Destination Gain", 0.0));

template <typename T>
class BufferComposeClient : public FluidBaseClient<decltype(BufComposeParams), BufComposeParams>, OfflineIn, OfflineOut
{
  using HostVector = FluidTensorView<T, 1>;
  using HostMatrix = FluidTensor<T, 2>;

public:
  BufferComposeClient(ParamSetViewType &p) : FluidBaseClient(p)
  {}

  Result process()
  {

    if (!get<kSource>().get()) { return {Result::Status::kError, "No input"}; }

    size_t nChannels{0};
    size_t nFrames{0};

    {
      BufferAdaptor::Access source(get<kSource>().get());

      if (!(source.exists() && source.valid())) return {Result::Status::kError, "Source Buffer Not Found or Invalid"};

      nChannels = get<kNChans>() == -1 ? source.numChans() - get<kStartChan>() : get<kNChans>();
      nFrames   = get<kNumFrames>() == -1 ? source.numFrames() - get<kOffset>() : get<kNumFrames>();

      if (nChannels <= 0 || nFrames <= 0) return {Result::Status::kError, "Zero length segment requested"};

      // We don't care if the overall number of frames will overrun, because we'll zero pad, but the
      // offset should be within the source range
      if (get<kOffset>() >= source.numFrames())
        return {Result::Status::kError, "Start frame (", get<kOffset>(), ") out of range."};

      // We don't care if the overall number of channels will overrun, because we'll loop, but the
      // offset should be within the source range
      if (get<kStartChan>() >= source.numChans())
        return {Result::Status::kError, "Start channel ", get<kStartChan>(), " out of range."};
    }

    auto       dstStart     = get<kDestOffset>();
    auto       dstStartChan = get<kDestStartChan>();
    auto       dstEnd{0};
    auto       dstEndChan{0};
    bool       destinationResizeNeeded{false};
    HostMatrix destinationOrig(0, 0);

    {
      BufferAdaptor::Access destination(get<kDest>().get());

      if (!destination.exists())
        return {Result::Status::kError, "Destination Buffer Not Found or Invalid"};

      dstEnd     = dstStart + nFrames ;
      dstEndChan = dstStartChan + nChannels;

      destinationResizeNeeded = (dstEnd > destination.numFrames()) || (dstEndChan > destination.numChans());

        auto applyGain = [this](T&x){x *= get<kDestGain>();};


      if (destinationResizeNeeded) // copy the whole of desintation if we have to resize it
      {
        destinationOrig.resize(std::max<unsigned>(dstEndChan, destination.numChans()), std::max<unsigned>(dstEnd,destination.numFrames()));
        if(destination.numChans() > 0 && destination.numFrames() > 0)
        {
            for (int i = 0; i < destination.numChans(); ++i)
                destinationOrig.row(i)(Slice(0, destination.numFrames())) = destination.samps(i);
            destinationOrig(Slice(dstStartChan, dstEndChan - dstStartChan), Slice(dstStart, dstEnd - dstStart)).apply(applyGain);
        }

      } else // just copy what we're affecting
      {
        destinationOrig.resize(nChannels, nFrames);
        for (int i = 0; i < nChannels; ++i)
        {
          destinationOrig.row(i) = destination.samps(dstStart, nFrames, dstStartChan + i);
          destinationOrig.row(i).apply(applyGain);
        }
      }
    }

    // Access sources inside own scope block, so they'll be unlocked before
    // we need to (possibly) resize the desintation buffer which could
    // (possibly)  also be one of the sources
    {
      BufferAdaptor::Access source(get<kSource>().get());
      auto                  gain = get<kGain>();
      // iterates through the copying of the first source
      for (size_t i = dstStartChan, j = 0; j < nChannels; ++i, ++j)
      {
        // Special repeating channel voodoo
        HostVector sourceChunk{
            source.samps(get<kOffset>(), std::min(nFrames, source.numFrames()), (get<kStartChan>() + j) % source.numChans())};

        HostVector destinationChunk{destinationOrig.row(j)};
        if (destinationResizeNeeded) { destinationChunk.reset(destinationOrig.row(i).data(), dstStart, nFrames); }

        std::transform(sourceChunk.begin(), sourceChunk.end(), destinationChunk.begin(), destinationChunk.begin(),
                       [gain](T &src, T &dst) { return dst + src * gain; });
      }
    }

    BufferAdaptor::Access destination(get<kDest>().get());

    if (destinationResizeNeeded)
    {
      destination.resize(destinationOrig.cols(), destinationOrig.rows(), 1, destination.sampleRate());
      for (int i = 0; i < destination.numChans(); ++i) destination.samps(i) = destinationOrig.row(i);
    } else
    {
      for (int i = 0; i < nChannels; ++i)
        destination.samps(dstStart, nFrames, dstStartChan + i) = destinationOrig.row(i);
    }

    return {Result::Status::kOk};
  }
};
} // namespace client
} // namespace fluid
