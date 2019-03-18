#pragma once

#include <clients/common/BufferAdaptor.hpp>

namespace fluid {
namespace client {
struct Offline {};

struct OfflineIn: Offline{};
struct OfflineOut:Offline{};

struct BufferProcessSpec
{
  BufferProcessSpec() = default;

  BufferProcessSpec(BufferAdaptor* b, intptr_t o, intptr_t nf, intptr_t co, intptr_t nc):
  buffer{b},startFrame{o}, nFrames{nf}, startChan{co}, nChans{nc}
  {}

  BufferAdaptor* buffer;
  intptr_t startFrame = 0;
  intptr_t nFrames = -1;
  intptr_t startChan = 0; 
  intptr_t nChans = -1;
};


} //namespace client
}//namespace fluid
