#pragma once

#include "BufferAdaptor.hpp"

namespace fluid {
namespace client {
        
class MemoryBufferAdaptor: public BufferAdaptor
{
public:
    
   MemoryBufferAdaptor(size_t chans, size_t frames, double sampleRate) : mData(chans,frames)
   {}

   // N.B. -cannot copy const BufferAdaptors at the moment
    
   MemoryBufferAdaptor(BufferAdaptor& other) { *this = other; }

  // N.B.  -cannot get access to a const BufferAdaptor at the moment

   MemoryBufferAdaptor& operator=(BufferAdaptor& other)
   {
      if(this != &other)
      {
        BufferAdaptor::Access src(&other);
        mData.resize(src.numChans() * src.rank(),src.numFrames());
        mExists = src.exists();
        mValid = src.valid();
        mRank = src.rank();
        mSampleRate = src.sampleRate();
        for(size_t i = 0; i < numChans() * mRank; i++)
          mData.row(i) = src.samps(i / mRank, i % mRank);
        mOrigin = &other;
      }
      return *this;
   }

  ~MemoryBufferAdaptor()
  {
    if(mOrigin)
    {
      BufferAdaptor::Access src(mOrigin);
      if(src.exists() && src.valid())
      {
        if(numChans() != src.numChans() || numFrames() != src.numFrames() || mRank != src.rank())
          src.resize(numFrames(),numChans(),mRank,mSampleRate);
        
        for(int i = 0; i < numChans(); ++i)
          for(int j = 0; j < mRank; ++j)
            src.samps(i / mRank,j) = samps(i / mRank,j);
      }
      //TODO feedback failure to user somehow: I need a message queue
      return;

    }
  }

   bool acquire() override { return true; }
   void release() override {}
   bool valid() const override { return mValid; }
   bool exists() const override  { return mExists; }
    
   void resize(size_t frames, size_t channels, size_t rank, double sampleRate) override
   {
     mRank = rank;
     mSampleRate = sampleRate;
     mData.resize(channels * rank, frames);
   }
    
   // Return a slice of the buffer
   FluidTensorView<float, 1> samps(size_t channel, size_t rankIdx = 0) { return mData.col(channel * mRank + rankIdx); }
   FluidTensorView<float, 1> samps(size_t offset, size_t nframes, size_t chanoffset) { return mData(Slice(offset, nframes), Slice(chanoffset, 1)).row(0); }
   size_t numFrames() const override { return mData.cols(); }
   size_t numChans() const override { return mRank ? (mData.rows() / mRank) : 0; }
   size_t rank() const override { return mRank; }
   double sampleRate() const { return mSampleRate; }
  
  private:
    BufferAdaptor* mOrigin;
    FluidTensor<float, 2> mData;
    double mSampleRate;
    bool mValid{true};
    bool mExists{true};
    int mRank{1};
};
    
}
}
