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

#include "../common/FluidTask.hpp"
#include "../common/Result.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"

#include "../../algorithms/util/FFT.hpp"

namespace fluid {
namespace client {


class FluidContext
{
public:
  //  addError()
  FluidContext() = default;
  FluidContext(FluidTask& t) : mTask{&t} {}
  FluidContext(index vectorSize, Allocator& alloc):
      mVectorSize{vectorSize},  mAllocator{&alloc}
  {};
  
  FluidTask* task() { return mTask; }
  void       task(FluidTask* t) { mTask = t; }
  
  Allocator& allocator() const noexcept
  {
    return mAllocator ? *mAllocator : FluidDefaultAllocator();
  }
  
  index hostVectorSize() const noexcept {
    return mVectorSize;
  }
  
  void hostVectorSize(index vs) { mVectorSize = vs; };
  
private:  
  FluidTask*  mTask{nullptr};
  index mVectorSize{0};
  Allocator*  mAllocator{nullptr};
  MessageList mMessages;
};

} // namespace client
} // namespace fluid
