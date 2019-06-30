#pragma once

#include <data/TensorTypes.hpp>
#include "Result.hpp"

namespace fluid {
namespace client {
class BufferAdaptor
{
public:
  class Access
  {
  public:
    Access(BufferAdaptor *adaptor) : mAdaptor(nullptr)
    {
      if (adaptor && adaptor->acquire())
        mAdaptor = adaptor;
    }

    ~Access()
    {
      if (mAdaptor) mAdaptor->release();
    }

    Access(const Access &) = delete;
    Access &operator=(const Access &) = delete;
    Access(Access &&)                 = default;
    Access &operator=(Access &&) = default;

    void destroy()
    {
      if (mAdaptor) mAdaptor->release();
      mAdaptor = nullptr;
    }

    bool valid() const { return mAdaptor ? mAdaptor->valid() : false; }

    bool exists() const { return mAdaptor ? mAdaptor->exists() : false; }

    void resize(size_t frames, size_t channels, double sampleRate)
    {
      if (mAdaptor) mAdaptor->resize(frames, channels, sampleRate);
    }

    FluidTensorView<float, 1> samps(size_t channel)
    {
      assert(mAdaptor);
      return mAdaptor->samps(channel);
    }

    FluidTensorView<float, 1> samps(size_t offset, size_t nframes, size_t chanoffset)
    {
      assert(mAdaptor);
      return mAdaptor->samps(offset, nframes, chanoffset);
    }
      
    size_t numFrames() const { return mAdaptor ? mAdaptor->numFrames() : 0; }

    size_t numChans() const { return mAdaptor ? mAdaptor->numChans() : 0; }

    double sampleRate() const { return mAdaptor ? mAdaptor->sampleRate() : 0; }
  private:
    BufferAdaptor *mAdaptor;
  };

  BufferAdaptor(BufferAdaptor &&rhs) = default;
  BufferAdaptor()                    = default;

  virtual ~BufferAdaptor()
  {
    //      destroy();
  }

private:
  virtual bool acquire()                                           = 0;
  virtual void release()                                           = 0;
  virtual bool valid() const                                       = 0;
  virtual bool exists() const                                      = 0;
  virtual void resize(size_t frames, size_t channels, double sampleRate) = 0;
  // Return a slice of the buffer
  virtual FluidTensorView<float, 1> samps(size_t channel)               = 0;
    
  virtual FluidTensorView<float, 1> samps(size_t offset, size_t nframes, size_t chanoffset) = 0;
  virtual size_t                    numFrames() const                                       = 0;
  virtual size_t                    numChans() const                                        = 0;
  virtual double                    sampleRate() const                                      = 0;
};

Result bufferRangeCheck(BufferAdaptor* b, intptr_t startFrame, intptr_t& nFrames, intptr_t startChan, intptr_t& nChans)
{
    if(!b)
      return {Result::Status::kError, "Input buffer not set"}; //error

    BufferAdaptor::Access thisInput(b);

    if(!thisInput.exists())
      return {Result::Status::kError, "Input buffer ", b, " not found."} ; //error

    if(!thisInput.valid())
      return {Result::Status::kError, "Input buffer ", b, "invalid (possibly zero-size?)"} ; //error

    if(startFrame >= thisInput.numFrames() || startFrame < 0)
      return {Result::Status::kError, "Input buffer ", b, "invalid start frame ", startFrame}; //error

    if(startChan >= thisInput.numChans() || startChan < 0)
      return {Result::Status::kError, "Input buffer ", b, "invalid start channel ", startChan}; //error

    nFrames = nFrames < 0 ? thisInput.numFrames() - startFrame: nFrames;
    if(nFrames <= 0)
      return {Result::Status::kError, "Input buffer ", b, ": not enough frames" }; //error

    nChans = nChans < 0 ? thisInput.numChans() - startChan : nChans;
    if(startChan <= 0)
      return {Result::Status::kError, "Input buffer ", b, ": not enough channels" }; //error

   return {Result::Status::kOk,""};
}


} // namespace client
} // namespace fluid

