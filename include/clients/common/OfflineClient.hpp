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

  BufferProcessSpec(BufferAdaptor* b, int o, int nf, int co, int nc):
  buffer{b},startFrame{o}, nFrames{nf}, startChan{co}, nChans{nc}
  {}

  BufferAdaptor* buffer;
  long startFrame = 0;
  long nFrames = -1;
  long startChan = 0; 
  long nChans = -1;
};


} //namespace client
}//namespace fluid
