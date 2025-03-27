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

#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/FluidTensor.hpp"
#include <cassert>

namespace fluid {

/// Input buffer, with possibly overlapped reads
template <typename T>
class FluidSource
{
  using Matrix = FluidTensor<T, 2>;
  using View = FluidTensorView<T, 2>;

//  using Container = rt::vector<T>;

public:
  FluidSource(const FluidSource&) = delete;
  FluidSource& operator=(const FluidSource&) = delete;
  FluidSource(FluidSource&&) noexcept = default;
  FluidSource& operator=(FluidSource&&) noexcept = default;

  FluidSource(const index size, const index channels, index maxHostBufferSize,
              Allocator& alloc = FluidDefaultAllocator())
      : mSize(size), mChannels(channels), mHostBufferSize(maxHostBufferSize),
        mMaxHostBufferSize(maxHostBufferSize),
        matrix(channels, bufferSize(), alloc)
  {}

  FluidSource() : FluidSource(0, 1, 0, FluidDefaultAllocator()){};


  template <typename U>
  void push(const std::vector<FluidTensorView<U, 1>>& in)
  {
    static_assert(std::is_convertible<U, T>::value,
                  "Can't convert between types");

    assert(in.size() == asUnsigned(mChannels));
    index blocksize = in[0].size();
    doPush(in, blocksize);
  }

  template <typename U>
  void push(FluidTensorView<U, 2> in)
  {
    static_assert(std::is_convertible<U, T>::value,
                  "Can't convert between types");

    assert(in.rows() == mChannels);
    index blocksize = in.cols();
    doPush(in, blocksize);
  }

  /// Pull a frame of data out of the buffer.
  void pull(View out, index frameTime)
  {
    index blocksize = out.cols();
    index offset = mHostBufferSize - frameTime;

    if (offset > bufferSize())
    {
      out.fill(0);
      return;
    }

    offset += blocksize;
    offset = (offset <= mCounter) ? mCounter - offset
                                  : mCounter + bufferSize() - offset;

    index size =
        (offset + blocksize > bufferSize()) ? bufferSize() - offset : blocksize;

    out(Slice(0), Slice(0, size)) <<= matrix(Slice(0), Slice(offset, size));
    out(Slice(0), Slice(size, blocksize - size)) <<=
        matrix(Slice(0), Slice(0, blocksize - size));
  }

  void setHostBufferSize(const index size)
  {
    assert(size <= mMaxHostBufferSize);
    mHostBufferSize = size;
  }

  /// Reset the buffer, resizing if the desired
  /// size and / or host buffer size have changed
  void reset()
  {
    std::fill(matrix.begin(), matrix.end(), 0);
    mCounter = 0;
  }


  index channels() const noexcept { return mChannels; }
  index size() const noexcept { return mSize; }
  index hostBufferSize() const noexcept { return mHostBufferSize; }

private:
  index bufferSize() const { return mSize + mHostBufferSize; }


  template <typename U>
  void doPush(U& in, index blocksize)
  {
    assert(blocksize <= bufferSize());
    index offset = mCounter;
    index size = ((offset + blocksize) > bufferSize()) ? bufferSize() - offset
                                                       : blocksize;
    index overspill = blocksize - size;
    copyContainer(in, offset, size, overspill);
  }

  template <typename U>
  void copyContainer(FluidTensorView<U, 2> x, index offset, index size,
                     index overspill)
  {
    // Copy all channels (rows)
    copyIn(x(Slice(0), Slice(0, size)), offset, size, Slice(0));
    copyIn(x(Slice(0), Slice(size, overspill)), 0, overspill, Slice(0));
  }

  template <typename U>
  void copyContainer(const std::vector<FluidTensorView<U, 1>>& x, index offset,
                     index size, index overspill)
  {
    for (index i = 0; i < mChannels; ++i)
      copyIn(FluidTensorView<U, 2>(x[asUnsigned(i)](Slice(0, size))), offset,
             size, i, i == mChannels - 1);
    for (index i = 0; i < mChannels; ++i)
      copyIn(FluidTensorView<U, 2>(x[asUnsigned(i)](Slice(size, overspill))), 0,
             overspill, i, i == mChannels - 1);
  }

  template <typename U, typename Chans>
  void copyIn(FluidTensorView<U, 2> input, index offset, index size,
              Chans chans, bool incrementTime = true)
  {
    if (size)
    {
      matrix(chans, Slice(offset, size)) <<= input;
      if (incrementTime) mCounter = offset + size;
    }
  }

  index     mCounter = 0;
  index     mSize;
  index     mChannels;
  index     mHostBufferSize = 0;
  index     mMaxHostBufferSize = 0;
  Matrix    matrix;
};
} // namespace fluid
