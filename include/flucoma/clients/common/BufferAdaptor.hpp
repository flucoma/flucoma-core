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

#include "Result.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/TensorTypes.hpp"

namespace fluid {
namespace client {
class BufferAdaptor
{
public:
  class ReadAccess
  {
  public:
    ReadAccess(const BufferAdaptor* adaptor) : mAdaptor(nullptr)
    {
      if (adaptor && adaptor->acquire()) mAdaptor = adaptor;
    }

    ~ReadAccess()
    {
      if (mAdaptor) mAdaptor->release();
    }

    ReadAccess(const ReadAccess&) = delete;
    ReadAccess& operator=(const ReadAccess&) = delete;
    ReadAccess(ReadAccess&&) noexcept = default;
    ReadAccess& operator=(ReadAccess&&) noexcept = default;

    void destroy()
    {
      if (mAdaptor) mAdaptor->release();
      mAdaptor = nullptr;
    }

    bool valid() const { return mAdaptor ? mAdaptor->valid() : false; }

    bool exists() const { return mAdaptor ? mAdaptor->exists() : false; }

    FluidTensorView<const float, 2> allFrames() const
    {
      assert(mAdaptor);
      return mAdaptor->allFrames();
    }

    FluidTensorView<const float, 1> samps(index channel) const
    {
      assert(mAdaptor);
      return mAdaptor->samps(channel);
    }

    FluidTensorView<const float, 1> samps(index offset, index nframes,
                                          index chanoffset) const
    {
      assert(mAdaptor);
      return mAdaptor->samps(offset, nframes, chanoffset);
    }

    index numFrames() const { return mAdaptor ? mAdaptor->numFrames() : 0; }

    index numChans() const { return mAdaptor ? mAdaptor->numChans() : 0; }

    double sampleRate() const { return mAdaptor ? mAdaptor->sampleRate() : 0; }

  private:
    const BufferAdaptor* mAdaptor;
  };

  class Access : public ReadAccess
  {
  public:
    Access(BufferAdaptor* adaptor)
        : ReadAccess(adaptor), mMutableAdaptor{adaptor}
    {}

    // Force any needed refreshing of mutable buffers (if the client class
    // overrides refresh())
    ~Access()
    {
      if (mMutableAdaptor) mMutableAdaptor->refresh();
    }

    Access(const Access&) = delete;
    Access& operator=(const Access&) = delete;
    Access(Access&&) noexcept = default;
    Access& operator=(Access&&) noexcept = default;

    FluidTensorView<float, 2> allFrames()
    {
      assert(mMutableAdaptor);
      return mMutableAdaptor->allFrames();
    }

    FluidTensorView<float, 1> samps(index channel)
    {
      assert(mMutableAdaptor);
      return mMutableAdaptor->samps(channel);
    }

    FluidTensorView<float, 1> samps(index offset, index nframes,
                                    index chanoffset)
    {
      assert(mMutableAdaptor);
      return mMutableAdaptor->samps(offset, nframes, chanoffset);
    }

    const Result resize(index frames, index channels, double sampleRate)
    {
      return mMutableAdaptor
                 ? mMutableAdaptor->resize(frames, channels, sampleRate)
                 : Result{Result::Status::kError,
                          "Trying to resize null buffer"};
    }

    void refresh()
    {
      if (mMutableAdaptor) mMutableAdaptor->refresh();
    }

  private:
    BufferAdaptor* mMutableAdaptor;
  };


  BufferAdaptor() = default;

  BufferAdaptor(BufferAdaptor&& rhs) noexcept = default;
  BufferAdaptor& operator=(BufferAdaptor&& rhs) noexcept = default;

  virtual ~BufferAdaptor()
  {
    //      destroy();
  }

private:
  virtual bool         acquire() const = 0;
  virtual void         release() const = 0;
  virtual bool         valid() const = 0;
  virtual bool         exists() const = 0;
  virtual const Result resize(index frames, index channels,
                              double sampleRate) = 0;
  virtual std::string  asString() const = 0;
  // Return a slice of the buffer
  virtual FluidTensorView<float, 1>       samps(index channel) = 0;
  virtual FluidTensorView<float, 1>       samps(index offset, index nframes,
                                                index chanoffset) = 0;
  virtual FluidTensorView<const float, 1> samps(index channel) const = 0;
  virtual FluidTensorView<const float, 1> samps(index offset, index nframes,
                                                index chanoffset) const = 0;

  virtual FluidTensorView<float, 2>       allFrames() = 0;
  virtual FluidTensorView<const float, 2> allFrames() const = 0;

  virtual index        numFrames() const = 0;
  virtual index        numChans() const = 0;
  virtual double       sampleRate() const = 0;
  virtual void         refresh(){};
  friend std::ostream& operator<<(std::ostream& os, const BufferAdaptor* b);
};

inline std::ostream& operator<<(std::ostream& os, const BufferAdaptor* b)
{
  return os << b->asString();
}

inline Result bufferRangeCheck(const BufferAdaptor* b, index startFrame,
                        index& nFrames, index startChan, index& nChans)
{
  if (!b) return {Result::Status::kError, "Input buffer not set"}; // error

  BufferAdaptor::ReadAccess thisInput(b);

  if (!thisInput.exists())
    return {Result::Status::kError, "Input buffer ", b, " not found."}; // error

  if (!thisInput.valid())
    return {Result::Status::kError, "Input buffer ", b,
            " invalid (possibly zero-size?)"}; // error

  if (startFrame >= thisInput.numFrames() || startFrame < 0)
    return {Result::Status::kError, "Input buffer ", b, " invalid start frame ",
            startFrame}; // error

  if (startChan >= thisInput.numChans() || startChan < 0)
    return {Result::Status::kError, "Input buffer ", b,
            " invalid start channel ", startChan}; // error

  nFrames = nFrames < 0 ? thisInput.numFrames() - startFrame : nFrames;
  if (nFrames <= 0 || nFrames > thisInput.numFrames() - startFrame)
    return {Result::Status::kError, "Input buffer ", b,
            ": not enough frames"}; // error

  nChans = nChans < 0 ? thisInput.numChans() - startChan : nChans;
  if (nChans <= 0 || nChans > thisInput.numChans() - startChan)
    return {Result::Status::kError, "Input buffer ", b,
            ": not enough channels"}; // error

  return {Result::Status::kOk, ""};
}


} // namespace client
} // namespace fluid
