#pragma once

#include "BufferAdaptor.hpp"

namespace fluid {
namespace client {
        
class MemoryBufferAdaptor: public BufferAdaptor
{
public:
    
   MemoryBufferAdaptor(size_t chans, size_t frames, double sampleRate) : mData(frames,chans)
   {}

   // N.B. -cannot copy const BufferAdaptors at the moment
    
  MemoryBufferAdaptor(std::shared_ptr<BufferAdaptor>& other) { *this = other; }

  // N.B.  -cannot get access to a const BufferAdaptor at the moment

   MemoryBufferAdaptor& operator=(std::shared_ptr<BufferAdaptor>& other)
   {
      if(this != other.get())
      {
        BufferAdaptor::Access src(other.get());
        mData.resize(src.numFrames(),src.numChans());
        mExists = src.exists();
        mValid = src.valid();
        mSampleRate = src.sampleRate();
        for(size_t i = 0; i < mData.cols(); i++)
          mData.col(i) = src.samps(0, src.numFrames(), i);
        mOrigin = other;
        mWrite = false;
      }
      return *this;
   }

  void copyToOrigin()
  {
    if(mWrite && mOrigin)
    {
      BufferAdaptor::Access src(mOrigin.get());
      if(src.exists())
      {
        if(numChans() != src.numChans() || numFrames() != src.numFrames())
          src.resize(numFrames(),numChans(),mSampleRate);
        
        if(src.valid())
          for(int i = 0; i < numChans(); ++i)
              src.samps(i) = samps(i);
      }
      //TODO feedback failure to user somehow: I need a message queue
      
    }
  }

   bool acquire() override { return true; }
   void release() override {}
   bool valid() const override { return mValid; }
   bool exists() const override  { return mExists; }
    
   void resize(size_t frames, size_t channels, double sampleRate) override
   {
     mWrite = true;
     mSampleRate = sampleRate;
     mData.resize(frames,channels);
   }
    
   // Return a slice of the buffer
   FluidTensorView<float, 1> samps(size_t channel) { return mData.col(channel); }
   FluidTensorView<float, 1> samps(size_t offset, size_t nframes, size_t chanoffset) { return mData(Slice(offset, nframes), Slice(chanoffset, 1)).col(0); }
   size_t numFrames() const override { return mData.rows(); }
   size_t numChans() const override { return mData.cols(); }
   double sampleRate() const { return mSampleRate; }
   void refresh() override { mWrite = true; }
  private:
    std::shared_ptr<BufferAdaptor> mOrigin;
    FluidTensor<float, 2> mData;
    double mSampleRate;
    bool mValid{true};
    bool mExists{true};
    bool mWrite{false};
};
    
}
}
