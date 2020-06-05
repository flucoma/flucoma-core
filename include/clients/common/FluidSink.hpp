/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
#pragma once

#include "../../data/FluidIndex.hpp"
#include "../../data/FluidTensor.hpp"
#include <cassert>

namespace fluid {

/*
 FluidSink

 An output buffer, with overlap-add
 */
template <typename T>
class FluidSink
{
  using tensor_type = FluidTensor<T, 2>;
  using view_type = FluidTensorView<T, 2>;
  using const_view_type = const FluidTensorView<T, 2>;

public:
  FluidSink() : FluidSink(0, 1) {}

  FluidSink(const FluidSink&) = delete;
  FluidSink& operator=(const FluidSink&) = delete;
  FluidSink(FluidSink&&) noexcept = default;
  FluidSink& operator=(FluidSink&&) noexcept = default;

  FluidSink(const index size, const index channels = 1)
      : matrix(channels, size), mSize(size), mChannels(channels)
  {}

  tensor_type& data() { return matrix; }

  /**
   Accumulate data into the buffer, optionally moving
   the write head on by a custom amount.

   This *adds* the content of the frame to whatever is
   already there
   **/
  void push(const_view_type x, index frameTime)
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

  /**
   Copy data from the buffer, and zero where it was
   **/
  void pull(view_type out)
  {

    index blocksize = out.cols();
    if (blocksize > bufferSize()) { return; }

    index offset = mCounter;

    index size =
        offset + blocksize > bufferSize() ? bufferSize() - offset : blocksize;

    outAndZero(out(Slice(0), Slice(0, size)), offset, size);
    outAndZero(out(Slice(0), Slice(size, blocksize - size)), 0,
               blocksize - size);
  }

  /*!
   Reset the buffer, resizing if the host buffer size
   or user buffer size have changed.

   This should be called from an audio host's DSP setup routine
   **/
  void reset(index channels = 0)
  {
    if (channels) mChannels = channels;

    if (matrix.cols() != bufferSize() || matrix.rows() != channels)
    {
      matrix.resize(mChannels, bufferSize());
    }
    matrix.fill(0);
    mCounter = 0;  
  }

  void setSize(index n) { mSize = n; }

  void setHostBufferSize(index n) { mHostBufferSize = n; }

  index channels() const noexcept { return mChannels; }
  index size() const noexcept { return mSize; }
  index hostBufferSize() const noexcept { return mHostBufferSize; }

private:
  void addIn(const_view_type in, index offset, index size)
  {
    if (size)
    {
      matrix(Slice(0), Slice(offset, size)).apply(in, [](double& x, double y) {
        x += y;
      });
    }
  }

  void outAndZero(view_type out, index offset, index size)
  {
    if (size)
    {
      view_type buf = matrix(Slice(0), Slice(offset, size));
      view_type output = buf;
      out = output;
      buf.fill(0);
      mCounter = offset + size;
    }
  }

  template <typename OutputIt>
  void outAndZero(OutputIt out, OutputIt end, index outOffset, index offset,
                  index size)
  {
    if (size)
    {
      for (index i = 0; (i < mChannels && out != end); ++i, ++out)
      {
        auto outSlice = matrix(++i, Slice(offset, size));
        *out->copy_to(outSlice, outOffset, size);
      }
      matrix(Slice(0), Slice(offset, size)).fill(0);
      mCounter = offset + size;
    }
  }

  index bufferSize() const { return mSize + mHostBufferSize; }

  tensor_type matrix;
  index       mSize;
  index       mChannels;
  index       mCounter = 0;
  index       mHostBufferSize = 0;
};
} // namespace fluid
