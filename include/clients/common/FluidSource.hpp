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

/*!
 FluidSource

 Input buffer, with possibly overlapped reads
 */
template <typename T>
class FluidSource //: public FluidTensor<T,2>
{
  using tensor_type = FluidTensor<T, 2>;
  using view_type = FluidTensorView<T, 2>;
  using const_view_type = const FluidTensorView<T, 2>;

public:
  FluidSource(const FluidSource&) = delete;
  FluidSource& operator=(const FluidSource&) = delete;
  FluidSource(FluidSource&&) noexcept = default;
  FluidSource& operator=(FluidSource&&) noexcept = default;

  FluidSource(const index size, const index channels = 1)
      : matrix(channels, size), mSize(size), mChannels(channels)
  {}

  FluidSource() : FluidSource(0, 1){};

  tensor_type& data() { return matrix; }

  /*
   Push a frame of data into the buffer
   */
  void push(const_view_type x)
  {
    assert(x.rows() == mChannels);

    index blocksize = x.cols();

    assert(blocksize <= bufferSize());

    index offset = mCounter;

    index size = ((offset + blocksize) > bufferSize()) ? bufferSize() - offset
                                                       : blocksize;

    // Copy all channels (cols)
    copyIn(x(Slice(0), Slice(0, size)), offset, size);
    copyIn(x(Slice(0), Slice(size, blocksize - size)), 0, blocksize - size);
  }

  template <typename U>
  void push(FluidTensorView<U, 2> x)
  {
    static_assert(std::is_convertible<U, double>(),
                  "Can't convert between types");

    assert(x.rows() == mChannels);

    index blocksize = x.cols();

    assert(blocksize <= bufferSize());

    index offset = mCounter;

    index size = ((offset + blocksize) > bufferSize()) ? bufferSize() - offset
                                                       : blocksize;

    // Copy all channels (rows)
    copyIn(x(Slice(0), Slice(0, size)), offset, size);
    copyIn(x(Slice(0), Slice(size, blocksize - size)), 0, blocksize - size);
  }

  //  template <typename InputIt>
  //  void push(InputIt in, InputIt end, index nsamps, index nchans) {
  //    assert(nchans == mChannels);
  //    assert(nsamps <= bufferSize());
  //    index blocksize = nsamps;
  //
  //    index offset = mCounter;
  //
  //    index size = ((offset + blocksize) > bufferSize()) ? bufferSize() -
  //    offset
  //                                                        : blocksize;
  //
  //    copyIn(in, end, 0, offset, size);
  //    copyIn(in, end, size, 0, blocksize - size);
  //  }

  /*!
   Pull a frame of data out of the buffer.
   */
  void pull(view_type out, index frameTime)
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

    out(Slice(0), Slice(0, size)) = matrix(Slice(0), Slice(offset, size));
    out(Slice(0), Slice(size, blocksize - size)) =
        matrix(Slice(0), Slice(0, blocksize - size));
  }

  /*
   Set the buffer size of the enclosing host.
   Needed to properly handle latency, causality etc
   */
  void setHostBufferSize(const index size) { mHostBufferSize = size; }

  /*
   Reset the buffer, resizing if the desired
   size and / or host buffer size have changed

   This should be called in the DSP setup routine of
   the audio host
   */
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

  index channels() const noexcept { return mChannels; }
  index size() const noexcept { return mSize; }
  index hostBufferSize() const noexcept { return mHostBufferSize; }

private:
  /*
   Report the size of the whole buffer
   */
  index bufferSize() const { return mSize + mHostBufferSize; }

  //    /*
  //     Copy a frame into the buffer and move the write head on
  //     */
  //    void copy_in(const_view_type input, const index offset, const index
  //    size)
  //    {
  //      if(size)
  //      {
  //        matrix(slice(0),slice(offset,size)) = input;
  //        mCounter = offset + size;
  //      }
  //    }

  template <typename U>
  void copyIn(const FluidTensorView<U, 2> input, const index offset,
              const index size)
  {
    if (size)
    {
      matrix(Slice(0), Slice(offset, size)) = input;
      mCounter = offset + size;
    }
  }

  template <typename InputIt>
  void copyIn(InputIt in, InputIt end, const index inStart, const index offset,
              const index size)
  {
    if (size)
    {
      for (index i = 0; (i < mChannels && in != end); ++i, ++in)
      {
        auto inRange = matrix(i, Slice(offset, size));
        (*in)->copyFrom(inRange.row(0), inStart, size);
      }
      mCounter = offset + size;
    }
  }

  tensor_type matrix;
  index       mCounter = 0;
  index       mSize;
  index       mChannels;
  index       mHostBufferSize = 0;
};
} // namespace fluid
