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

#include "BufferAdaptor.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include <memory>

namespace fluid {
namespace client {

class MemoryBufferAdaptor : public BufferAdaptor
{
public:
  MemoryBufferAdaptor(index chans, index frames, double /*sampleRate*/)
      : mData(frames, chans)
  {}

  MemoryBufferAdaptor(std::shared_ptr<BufferAdaptor>& other) { *this = other; }
  MemoryBufferAdaptor(std::shared_ptr<const BufferAdaptor>& other)
  {
    *this = other;
  }

  MemoryBufferAdaptor& operator=(std::shared_ptr<BufferAdaptor>& other)
  {
    if (this != other.get())
    {
      *this = other.get();
      mOrigin = other;
    }
    return *this;
  }

  MemoryBufferAdaptor& operator=(std::shared_ptr<const BufferAdaptor>& other)
  {
    if (this != other.get()) *this = other.get();

    return *this;
  }

  void copyToOrigin(Result& r)
  {
    if (mWrite && mOrigin)
    {
      BufferAdaptor::Access src(mOrigin.get());
      if (src.exists())
      {
        if (numChans() != src.numChans() || numFrames() != src.numFrames())
          r = src.resize(numFrames(), numChans(), mSampleRate);

        if (r.ok() && src.valid())
          for (index i = 0; i < numChans(); ++i)
            src.samps(i)(Slice(0, numFrames())) <<= samps(i);
      }
      // TODO feedback failure to user somehow: I need a message queue
    }
  }

  bool acquire() const override { return true; }
  void release() const override {}
  bool valid() const override { return mValid; }
  bool exists() const override { return mExists; }

  const Result resize(index frames, index channels, double sampleRate) override
  {
    mWrite = true;
    mSampleRate = sampleRate;
    mData.resize(frames, channels);

    return Result();
  }

  FluidTensorView<float, 2> allFrames() override { return mData.transpose(); }

  FluidTensorView<const float, 2> allFrames() const override
  {
    FluidTensorSlice<2> tmp(mData.descriptor());
    return {tmp.transpose(), mData.data()};
  }

  // Return a slice of the buffer
  FluidTensorView<float, 1> samps(index channel) override
  {
    return mData.col(channel);
  }
  FluidTensorView<float, 1> samps(index offset, index nframes,
                                  index chanoffset) override
  {
    return mData(Slice(offset, nframes), Slice(chanoffset, 1)).col(0);
  }
  FluidTensorView<const float, 1> samps(index channel) const override
  {
    return mData.col(channel);
  }
  FluidTensorView<const float, 1> samps(index offset, index nframes,
                                        index chanoffset) const override
  {
    return mData(Slice(offset, nframes), Slice(chanoffset, 1)).col(0);
  }
  index       numFrames() const override { return mData.rows(); }
  index       numChans() const override { return mData.cols(); }
  double      sampleRate() const override { return mSampleRate; }
  std::string asString() const override { return ""; }
  void        refresh() override { mWrite = true; }

private:
  MemoryBufferAdaptor& operator=(const BufferAdaptor* other)
  {
    BufferAdaptor::ReadAccess src(other);    
    mExists = src.exists();
    mValid = src.valid();
    mSampleRate = src.sampleRate();
    mWrite = false;
    mOrigin = nullptr;
    if(mValid)
    {
      mData.resize(src.numFrames(), src.numChans());
      for (index i = 0; i < mData.cols(); i++)
        mData.col(i) <<= src.samps(0, src.numFrames(), i);
    }
    return *this;
  }

  std::shared_ptr<BufferAdaptor> mOrigin;
  FluidTensor<float, 2>          mData;
  double                         mSampleRate{44100};
  bool                           mValid{true};
  bool                           mExists{true};
  bool                           mWrite{false};
};

} // namespace client
} // namespace fluid
