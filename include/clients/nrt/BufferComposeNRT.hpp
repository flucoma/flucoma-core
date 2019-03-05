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
    BufferParam("srcBuf", "Source Buffer"), LongParam("startAt", "Source Offset", 0, Min(0)),
    LongParam("nFrames", "Source Number of Frames", -1), LongParam("startChan", "Source Channel Offset", 0, Min(0)),
    LongParam("nChans", "Source Number of Channels", -1), FloatParam("gain", "Source Gain", 1.0),
    BufferParam("dstBuf", "Destination Buffer"), LongParam("dstStartAt", "Destination Offset", 0),
    LongParam("dstStartChan", "Destination Channel Offset", 0), FloatParam("dstGain", "Destination Gain", 0.0));

template <typename Params, typename T, typename U>
class BufferComposeClient : public FluidBaseClient<Params>, OfflineIn, OfflineOut
{
  using HostVector = FluidTensorView<U, 1>;
  using HostMatrix = FluidTensor<U, 2>;

public:
  BufferComposeClient(Params &p)
      : mParams{p}
      , FluidBaseClient<Params>{p}
  {}

  Result process()
  {

    if (!param<kSource>(mParams).get()) { return {Result::Status::kError, "No input"}; }

    size_t nChannels{0};
    size_t nFrames{0};

    {
      BufferAdaptor::Access source(param<kSource>(mParams).get());

      if (!(source.exists() && source.valid())) return {Result::Status::kError, "Source Buffer Not Found or Invalid"};

      nChannels = param<kNChans>(mParams) == -1 ? source.numChans() - param<kStartChan>(mParams) : param<kNChans>(mParams);
      nFrames   = param<kNumFrames>(mParams) == -1 ? source.numFrames() - param<kOffset>(mParams) : param<kNumFrames>(mParams);

      if (nChannels <= 0 || nFrames <= 0) return {Result::Status::kError, "Zero length segment requested"};

      // We don't care if the overall number of frames will overrun, because we'll zero pad, but the
      // offset should be within the source range
      if (param<kOffset>(mParams) >= source.numFrames())
        return {Result::Status::kError, "Start frame (", param<kOffset>(mParams), ") out of range."};

      // We don't care if the overall number of channels will overrun, because we'll loop, but the
      // offset should be within the source range
      if (param<kStartChan>(mParams) >= source.numChans())
        return {Result::Status::kError, "Start channel ", param<kStartChan>(mParams), " out of range."};
    }

    auto       dstStart     = param<kDestOffset>(mParams);
    auto       dstStartChan = param<kDestStartChan>(mParams);
    auto       dstEnd{0};
    auto       dstEndChan{0};
    bool       destinationResizeNeeded{false};
    HostMatrix destinationOrig(0, 0);

    {
      BufferAdaptor::Access destination(param<kDest>(mParams).get());

      if (!destination.exists())
        return {Result::Status::kError, "Destination Buffer Not Found or Invalid"};

      dstEnd     = dstStart + nFrames ;
      dstEndChan = dstStartChan + nChannels;

      destinationResizeNeeded = (dstEnd > destination.numFrames()) || (dstEndChan > destination.numChans());
        
        auto applyGain = [this](U&x){x *= param<kDestGain>(mParams);};

        
      if (destinationResizeNeeded) // copy the whole of desintation if we have to resize it
      {
        destinationOrig.resize(std::max<unsigned>(dstEndChan, destination.numChans()), std::max<unsigned>(dstEnd,destination.numFrames()));
        if(destination.numChans() > 0 && destination.numFrames() > 0)
            for (int i = 0; i < destination.numChans(); ++i) {
                destinationOrig.row(i)(Slice(0, destination.numFrames())) = destination.samps(i);
                destinationOrig.row(i)(Slice(0, destination.numFrames())).apply(applyGain);
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
      BufferAdaptor::Access source(param<kSource>(mParams).get());
      auto                  gain = param<kGain>(mParams);
      // iterates through the copying of the first source
      for (size_t i = dstStartChan, j = 0; j < nChannels; ++i, ++j)
      {
        // Special repeating channel voodoo
        HostVector sourceChunk{
            source.samps(param<kOffset>(mParams), std::min(nFrames, source.numFrames()), (param<kStartChan>(mParams) + j) % source.numChans())};

        HostVector destinationChunk{destinationOrig.row(j)};
        if (destinationResizeNeeded) { destinationChunk.reset(destinationOrig.row(i).data(), dstStart, nFrames); }

        std::transform(sourceChunk.begin(), sourceChunk.end(), destinationChunk.begin(), destinationChunk.begin(),
                       [gain](U &src, U &dst) { return dst + src * gain; });
      }
    }

    BufferAdaptor::Access destination(param<kDest>(mParams).get());

    if (destinationResizeNeeded)
    {
      destination.resize(destinationOrig.cols(), destinationOrig.rows(), 1);
      for (int i = 0; i < destination.numChans(); ++i) destination.samps(i) = destinationOrig.row(i);
    } else
    {
      for (int i = 0; i < nChannels; ++i)
        destination.samps(dstStart, nFrames, dstStartChan + i) = destinationOrig.row(i);
    }

    return {Result::Status::kOk};
  }

private:
  Params &mParams;
};
} // namespace client
} // namespace fluid
