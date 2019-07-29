#pragma once

#include "Result.hpp"
#include "../../data/TensorTypes.hpp"

namespace fluid {
namespace client {
class BufferAdaptor
{
public:

  class ReadAccess
  {
  public:
    ReadAccess(const BufferAdaptor *adaptor) : mAdaptor(nullptr)
    {
      if (adaptor && adaptor->acquire())
        mAdaptor = adaptor;
    }

    ~ReadAccess()
    {
      if (mAdaptor) mAdaptor->release();
    }

    ReadAccess(const ReadAccess &) = delete;
    ReadAccess &operator=(const ReadAccess &) = delete;
    ReadAccess(ReadAccess &&) noexcept = default;
    ReadAccess &operator=(ReadAccess &&) noexcept = default;

    void destroy()
    {
      if (mAdaptor) mAdaptor->release();
      mAdaptor = nullptr;
    }

    bool valid() const { return mAdaptor ? mAdaptor->valid() : false; }

    bool exists() const { return mAdaptor ? mAdaptor->exists() : false; }

    const FluidTensorView<float, 1> samps(size_t channel) const
    {
      assert(mAdaptor);
      return mAdaptor->samps(channel);
    }

    const FluidTensorView<float, 1> samps(size_t offset, size_t nframes, size_t chanoffset) const
    {
      assert(mAdaptor);
      return mAdaptor->samps(offset, nframes, chanoffset);
    }
      
    size_t numFrames() const { return mAdaptor ? mAdaptor->numFrames() : 0; }

    size_t numChans() const { return mAdaptor ? mAdaptor->numChans() : 0; }

    double sampleRate() const { return mAdaptor ? mAdaptor->sampleRate() : 0; }
  private:
    const BufferAdaptor *mAdaptor;
  };
  
  class Access: public ReadAccess
  {
  public:
    Access(BufferAdaptor* adaptor): ReadAccess(adaptor), mMutableAdaptor{adaptor}
    {}
    
    //Force any needed refreshing of mutable buffers (if the client class overrides refresh())
    ~Access() { if(mMutableAdaptor) mMutableAdaptor->refresh(); }
    
    Access(const Access &) = delete;
    Access &operator=(const Access &) = delete;
    Access(Access &&) noexcept = default;
    Access &operator=(Access &&) noexcept = default;
      
    FluidTensorView<float, 1> samps(size_t channel)
    {
      assert(mMutableAdaptor);
      return mMutableAdaptor->samps(channel);
    }

    FluidTensorView<float, 1> samps(size_t offset, size_t nframes, size_t chanoffset)
    {
      assert(mMutableAdaptor);
      return mMutableAdaptor->samps(offset, nframes, chanoffset);
    }
    
    const Result resize(size_t frames, size_t channels, double sampleRate)
    {
      return mMutableAdaptor ?  mMutableAdaptor->resize(frames, channels, sampleRate) : Result{Result::Status::kError,"Trying to resize null buffer"};
    }
  
    private:
      BufferAdaptor* mMutableAdaptor;
  };
  
  BufferAdaptor(BufferAdaptor &&rhs) = default;
  BufferAdaptor()                    = default;

  virtual ~BufferAdaptor()
  {
    //      destroy();
  }

private:
  virtual bool acquire() const = 0;
  virtual void release() const = 0;
  virtual bool valid() const = 0;
  virtual bool exists() const = 0;
  virtual const Result resize(size_t frames, size_t channels, double sampleRate) = 0;
  virtual std::string asString() const = 0;
  // Return a slice of the buffer
  virtual FluidTensorView<float, 1> samps(size_t channel) = 0;
  virtual FluidTensorView<float, 1> samps(size_t offset, size_t nframes, size_t chanoffset) = 0;
  virtual const FluidTensorView<float, 1> samps(size_t channel)  const = 0;
  virtual const FluidTensorView<float, 1> samps(size_t offset, size_t nframes, size_t chanoffset) const = 0;
  virtual size_t numFrames() const = 0;
  virtual size_t numChans() const = 0;
  virtual double sampleRate() const = 0;
  virtual void refresh() {};
  friend std::ostream& operator<<(std::ostream& os, const BufferAdaptor* b); 
};

std::ostream& operator <<(std::ostream& os, const BufferAdaptor* b)
{
  return os << b->asString();
}
    
Result bufferRangeCheck(const BufferAdaptor* b, intptr_t startFrame, intptr_t& nFrames, intptr_t startChan, intptr_t& nChans)
{
    if(!b)
      return {Result::Status::kError, "Input buffer not set"}; //error

    BufferAdaptor::ReadAccess thisInput(b);

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
    if(nChans <= 0)
      return {Result::Status::kError, "Input buffer ", b, ": not enough channels" }; //error

   return {Result::Status::kOk,""};
}


} // namespace client
} // namespace fluid

