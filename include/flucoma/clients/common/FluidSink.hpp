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
#include <functional>

namespace fluid {

/// An output buffer, with overlap-add
template <typename T>
class FluidSink
{
//  using Container = rt::vector<T>;
  using Matrix = FluidTensor<T, 2>;
  using View = FluidTensorView<T, 2>;
  using const_view_type = const FluidTensorView<T, 2>;

public:
  FluidSink() : FluidSink(0, 1, 0) {}

  FluidSink(const FluidSink&) = delete;
  FluidSink& operator=(const FluidSink&) = delete;
  FluidSink(FluidSink&&) noexcept = default;
  FluidSink& operator=(FluidSink&&) noexcept = default;

  FluidSink(const index size, const index channels, index maxHostVectorSize,
            Allocator& alloc = FluidDefaultAllocator())
      : mSize(size), mChannels(channels), mHostBufferSize(maxHostVectorSize),
        mMaxHostBufferSize(maxHostVectorSize),
        matrix(channels, bufferSize(), alloc)
  {}

  /// Accumulate data into the buffer, optionally moving
  /// the write head on by a custom amount.

  /// This *adds* the content of the frame to whatever is
  /// already there
  void push(View x, index frameTime)
  {

    assert(x.rows() == mChannels);

    index blocksize = x.cols();

    assert(blocksize <= bufferSize());

    index offset = frameTime;

    if (offset + blocksize > bufferSize()) { return; }

    offset += mCounter;
    offset = offset < bufferSize() ? offset : offset - bufferSize();

    index size = ((offset + blocksize) > bufferSize()) ? bufferSize() - offset
                                                       : blocksize;

    addIn(x(Slice(0), Slice(0, size)), offset, size);
    addIn(x(Slice(0), Slice(size, blocksize - size)), 0, blocksize - size);
  }

  /// Copy data from the buffer, and zero where it was
  template <typename U>
  void pull(FluidTensorView<U, 2> out)
  {
    index blocksize = out.cols();
    if (blocksize > bufferSize()) { return; }
    doPull(out, blocksize);
  }

  template <typename U>
  void pull(const std::vector<FluidTensorView<U, 1>>& out)
  {
    assert(out.size() == mChannels);
    index blocksize = out[0].size();
    if (blocksize > bufferSize()) return;
    doPull(out, blocksize);
  }

  /// Reset the buffer, resizing if the host buffer size
  /// or user buffer size have changed.
  void reset()
  {
    std::fill(matrix.begin(), matrix.end(), 0);
    mCounter = 0;
  }

  void setHostBufferSize(index size)
  {
    assert(size <= mMaxHostBufferSize);
    mHostBufferSize = size;
  }

  index channels() const noexcept { return mChannels; }
  index size() const noexcept { return mSize; }
  index hostBufferSize() const noexcept { return mHostBufferSize; }

private:
  void addIn(const_view_type in, index offset, index size)
  {
    if (size)
    {
      matrix(Slice(0), Slice(offset, size)).apply(in, [](T& x, T y) {
        x += y;
      });
    }
  }

  template <typename U>
  void doPull(U& container, index blocksize)
  {
    index offset = mCounter;
    index size =
        offset + blocksize > bufferSize() ? bufferSize() - offset : blocksize;
    index overspill = blocksize - size;

    fillContainer(container, offset, size, overspill);
  }

  template <typename U>
  void fillContainer(FluidTensorView<U, 2> out, index offset, index size,
                     index overspill)
  {
    outAndZero(out(Slice(0), Slice(0, size)), offset, size, Slice(0));
    outAndZero(out(Slice(0), Slice(size, overspill)), 0, overspill, Slice(0));
  }

  template <typename U>
  void fillContainer(std::vector<FluidTensorView<U, 1>>& out, index offset,
                     index size, index overspill)
  {
    for (index i = 0; i < mChannels; ++i)
      outAndZero(FluidTensorView<U, 2>(out[0](Slice(0, size))), offset, size, i,
                 i == mChannels - 1);
    for (index i = 0; i < mChannels; ++i)
      outAndZero(FluidTensorView<U, 2>(out[0](Slice(size, overspill))), 0,
                 overspill, i, i == mChannels - 1);
  }

  template <typename U, typename Chans>
  void outAndZero(FluidTensorView<U, 2> out, index offset, index size,
                  Chans chans, bool incrementTime = true)
  {
    if (size)
    {
      View buf = matrix(chans, Slice(offset, size));
      View output = buf;
      out <<= output;
      buf.fill(0);
      if (incrementTime) mCounter = offset + size;
    }
  }

  index bufferSize() const { return mSize + mHostBufferSize; }

  index     mSize;
  index     mChannels;
  index     mCounter = 0;
  index     mHostBufferSize = 0;
  index     mMaxHostBufferSize = 0;
  Matrix      matrix;
};
} // namespace fluid
