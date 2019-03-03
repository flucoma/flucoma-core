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


//  bool validateBufferSpec(BufferProcessSpec& b)
//  {
//    BufferAdaptor::Access buf{b.buffer};
//    t_object* maxObj = (t_object*)static_cast<Wrapper*>(this);
//
//    const char* name = static_cast<MaxBufferAdaptor*>(b.buffer)->name()->s_name;
//
//    if(!buf.exists())
//    {
//      object_error(maxObj,"Buffer %s doesn't exist", name);
//      return false;
//    }
//
//    if(b.startFrame > buf.numFrames())
//    {
//      object_error(maxObj,"Buffer %s offset (%d) greater than size (%d) ", name, b.startFrame, buf.numFrames());
//      return false;
//    }
//
//    if(b.nFrames > 0 && b.startFrame + b.nFrames  > buf.numFrames())
//    {
//      object_error(maxObj,"Buffer %s offset plus length (%d) greater than size (%d) ", name, b.startFrame + b.nFrames, buf.numFrames());
//      return false;
//    }
//
//    if(b.startChan > buf.numChans())
//    {
//      object_error(maxObj,"Buffer %s offset channel (%d) greater than available channels (%d) ", name, b.startChan, buf.numChans());
//      return false;
//    }
//
//    if(b.nChans > 0 && b.startChan + b.nChans  > buf.numChans())
//    {
//      object_error(maxObj,"Buffer %s channel offset plus number of channels (%d) greater than available channels (%d) ", name, b.startChan + b.nChans, buf.numFrames());
//      return false;
//    }
//
//    return true;
//  }



} //namespace client
}//namespace fluid
