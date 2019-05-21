#pragma once

#include "BufferAdaptor.hpp"

class MemoryBufferAdaptor: public BufferAdaptor
{
   MemoryBufferAdaptor(int chans, int frames, double sampleRate): mData{chans,frames}
   {}

   MemoryBufferAdaptor(const BufferAdaptor& other) { *this = other; }

   MemoryBufferAdaptor& operator(const BufferAdaptor& other)
   {
      if(this != &other)
      {
        BufferAdaptor::Access src;
        mData.reszie(src.numChans(),src.numFrames());
        mExists = src.exists();
        mValid = src.Valid();
        mSampleRate = src.sampleRate();
        for(size_t i = 0; i < numChans())
          mData.row(i) = src.samps(i);
      }
      return *this;
   }

   bool acquire() override { return true; }
   void release() override {}
   bool valid()  override { return true; }
   bool exists() const override  { return true; }
   void resize(size_t frames, size_t channels, size_t rank, double sampleRate) override {}
  // Return a slice of the buffer
   FluidTensorView<float, 1> samps(size_t channel, size_t rankIdx = 0) { return mData.(Slice())
   FluidTensorView<float, 1> samps(size_t offset, size_t nframes, size_t chanoffset) = 0;
   size_t numFrames() const override { return mData.cols(); }
   size_t numChans() const  override { return mData.rows(); }
   size_t rank() const override {return mRank;}
   double sampleRate() const { return mSampleRate; }
  
  private:
    FluidTensor<float, 2> mData;
    double mSampleRate;
    bool mValid{true};
    bool mExists(true);
    int mRank{1};
}
