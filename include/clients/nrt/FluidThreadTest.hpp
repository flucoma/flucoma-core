#pragma once

#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterTypes.hpp"

#include <thread>
#include <chrono>

namespace fluid {
namespace client {

enum ThreadTestIdx { kResult, kWait};

auto constexpr ThreadTestParams = defineParameters(
    BufferParam("result","Output Result Buffer"),
    FloatParam("time", "Millisecond Wait", 0.0, Min(0.0))
);

template <typename T>
class ThreadTest
    : public FluidBaseClient<decltype(ThreadTestParams), ThreadTestParams>,
      public OfflineIn,
      public OfflineOut {

public:
  ThreadTest(ParamSetViewType &p) : FluidBaseClient(p) {}

  Result process(FluidContext& c) {
    using namespace std::chrono_literals;

    double wait = get<kWait>();

    for(auto i = 0; i < 1000; ++i)
    {
      std::this_thread::sleep_for(std::chrono::microseconds{static_cast<int>(wait)});
      if(!c.task()->processUpdate(i, 1000)) return {Result::Status::kCancelled,""};
    }

    if(auto bufref = get<kResult>())
    {
      auto buf = BufferAdaptor::Access(bufref.get());
      if(buf.exists())
      {
        if(!buf.resize(1,1,buf.sampleRate()).ok()) return {Result::Status::kError, "Buffer resize failed"};
        buf.samps(0)(0) = wait;
        return {Result::Status::kOk,""};
      } else return {Result::Status::kError, "Buffer not found"};
    }
    return {Result::Status::kError, "No buffer"};
  }
};

template <typename T>
using NRTThreadedThreadTest = NRTThreadingAdaptor<ThreadTest<T>>;

} // namespace client
} // namespace fluid
