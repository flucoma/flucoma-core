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
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterTypes.hpp"

namespace fluid {
namespace client {
namespace bufscale {

enum {
  kSource,
  kStartFrame,
  kNumFrames,
  kStartChan,
  kNumChans,
  kDest,
  kInLow,
  kInHigh,
  kOutLow,
  kOutHigh,
  kClip
};

constexpr auto BufScaleParams =
    defineParameters(InputBufferParam("source", "Source Buffer"),
                     LongParam("startFrame", "Source Offset", 0, Min(0)),
                     LongParam("numFrames", "Number of Frames", -1),
                     LongParam("startChan", "Start Channel", 0, Min(0)),
                     LongParam("numChans", "Number of Channels", -1),
                     BufferParam("destination", "Destination Buffer"),
                     FloatParam("inputLow", "Input Low Range", 0),
                     FloatParam("inputHigh", "Input High Range", 1),
                     FloatParam("outputLow", "Output Low Range", 0),
                     FloatParam("outputHigh", "Output High Range", 1),
                     EnumParam("clipping", "Optional Clipping", 0, "None",
                               "Minimum", "Maximum", "Both"));

class BufScaleClient : public FluidBaseClient,
                       public OfflineIn,
                       public OfflineOut
{

public:
  using ParamDescType = decltype(BufScaleParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return BufScaleParams; }

  BufScaleClient(ParamSetViewType& p, FluidContext&) : mParams(p) {}

  template <typename T>
  Result process(FluidContext&)
  {
    // retrieve the range requested and check it is valid
    index startFrame = get<kStartFrame>();
    index numFrames = get<kNumFrames>();
    index startChan = get<kStartChan>();
    index numChans = get<kNumChans>();

    Result r = bufferRangeCheck(get<kSource>().get(), startFrame, numFrames,
                                startChan, numChans);

    if (!r.ok()) return r;

    BufferAdaptor::ReadAccess source(get<kSource>().get());
    BufferAdaptor::Access     dest(get<kDest>().get());

    if (!dest.exists())
      return {Result::Status::kError, "Output buffer not found"};

    FluidTensor<double, 2> tmp(source.allFrames()(
        Slice(startChan, numChans), Slice(startFrame, numFrames)));

    // pre-process
    using clipFnType = double (*)(double, double, double);

    static const std::array<clipFnType, 4> clippingFunctions = {
        [](double x, double, double) { return x; },
        [](double x, double low, double) { return std::max(x, low); },
        [](double x, double, double high) { return std::min(x, high); },
        [](double x, double low, double high) {
          return std::min(std::max(x, low), high);
        }};

    auto   clipFn = clippingFunctions[asUnsigned(get<kClip>())];
    double outLow = get<kOutLow>();
    double outHigh = get<kOutHigh>();
    double scale = (outHigh - outLow) / (get<kInHigh>() - get<kInLow>());
    double offset = outLow - (scale * get<kInLow>());

    // process
    tmp.apply([&](double& x) {
      x *= scale;
      x += offset;
      x = clipFn(x, outLow, outHigh);
    });

    // write back the processed data, resizing the dest buffer
    r = dest.resize(numFrames, numChans, source.sampleRate());
    if (!r.ok()) return r;

    dest.allFrames() <<= tmp;

    return {};
  }
};
} // namespace bufscale

using NRTThreadedBufferScaleClient =
    NRTThreadingAdaptor<ClientWrapper<bufscale::BufScaleClient>>;
} // namespace client
} // namespace fluid
