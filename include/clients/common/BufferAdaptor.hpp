#pragma once

#include <data/TensorTypes.hpp>

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

    void resize(size_t frames, size_t channels, size_t rank, double sampleRate)
    {
      if (mAdaptor) mAdaptor->resize(frames, channels, rank, sampleRate);
    }

    FluidTensorView<float, 1> samps(size_t channel, size_t rankIdx = 0)
    {
      assert(mAdaptor);
      return mAdaptor->samps(channel, rankIdx);
    }

    FluidTensorView<float, 1> samps(size_t offset, size_t nframes, size_t chanoffset)
    {
      assert(mAdaptor);
      return mAdaptor->samps(offset, nframes, chanoffset);
    }
      
    size_t numFrames() const { return mAdaptor ? mAdaptor->numFrames() : 0; }

    size_t numChans() const { return mAdaptor ? mAdaptor->numChans() : 0; }

    size_t rank() const { return mAdaptor ? mAdaptor->rank() : 0; }

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
  virtual void resize(size_t frames, size_t channels, size_t rank, double sampleRate) = 0;
  // Return a slice of the buffer
  virtual FluidTensorView<float, 1> samps(size_t channel, size_t rankIdx = 0)               = 0;
  virtual FluidTensorView<float, 1> samps(size_t offset, size_t nframes, size_t chanoffset) = 0;
  virtual size_t                    numFrames() const                                       = 0;
  virtual size_t                    numChans() const                                        = 0;
  virtual size_t                    rank() const                                            = 0;
  virtual double                    sampleRate() const                                      = 0;
};
} // namespace client
} // namespace fluid

