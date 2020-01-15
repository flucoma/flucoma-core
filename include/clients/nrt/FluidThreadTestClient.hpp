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

#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterTypes.hpp"
#include <chrono>
#include <thread>

namespace fluid {
namespace client {

enum ThreadTestIdx { kResult, kWait };

auto constexpr ThreadTestParams =
    defineParameters(BufferParam("result", "Output Result Buffer"),
                     FloatParam("time", "Millisecond Wait", 0.0, Min(0.0)));

template <typename T>
class ThreadTestClient
    : public FluidBaseClient<decltype(ThreadTestParams), ThreadTestParams>,
      public OfflineIn,
      public OfflineOut
{

public:
  ThreadTestClient(ParamSetViewType& p) : FluidBaseClient(p) {}

  Result process(FluidContext& c)
  {
    using namespace std::chrono_literals;

    double wait = get<kWait>();

    for (auto i = 0; i < 1000; ++i)
    {
      std::this_thread::sleep_for(
          std::chrono::microseconds{static_cast<int>(wait)});
      if (!c.task()->processUpdate(i, 1000))
        return {Result::Status::kCancelled, ""};
    }

    if (auto bufref = get<kResult>())
    {
      auto buf = BufferAdaptor::Access(bufref.get());
      if (buf.exists())
      {
        if (!buf.resize(1, 1, buf.sampleRate()).ok())
          return {Result::Status::kError, "Buffer resize failed"};
        buf.samps(0)(0) = static_cast<float>(wait);
        return {Result::Status::kOk, ""};
      }
      else
        return {Result::Status::kError, "Buffer not found"};
    }
    return {Result::Status::kError, "No buffer"};
  }
};

template <typename T>
using NRTThreadedThreadTestClient = NRTThreadingAdaptor<ThreadTestClient<T>>;

} // namespace client
} // namespace fluid
