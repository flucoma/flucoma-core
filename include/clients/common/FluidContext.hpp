#pragma once

#include <clients/common/FluidTask.hpp>
#include <clients/common/Result.hpp>

namespace fluid {
namespace client {


class FluidContext
{

public:

//  addError()

  FluidContext(FluidTask& t): mTask{&t} {}
  FluidContext() = default;

  FluidTask* task() { return mTask; }
  void task(FluidTask* t) { mTask = t; }
private:
  FluidTask *mTask{nullptr};
  MessageList mMessages;
};

}
}
