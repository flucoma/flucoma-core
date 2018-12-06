/*!
 FluidBuffers.hpp

 Provide input and output buffering

 */
#pragma once

#include <data/FluidTensor.hpp>
#include <cassert>

namespace fluid {

/*
 FluidSink

 An output buffer, with overlap-add
 */
template <typename T> class FluidSink {
  using tensor_type = FluidTensor<T, 2>;
  using view_type = FluidTensorView<T, 2>;
  using const_view_type = const FluidTensorView<T, 2>;

public:
  FluidSink() : FluidSink(0, 1) {}

  FluidSink(FluidSink &) = delete;
  FluidSink operator=(FluidSink &) = delete;

  FluidSink(const size_t size, const size_t channels = 1)
      : matrix(channels,size), mSize(size), mChannels(channels) {}

  tensor_type &data() { return matrix; }

  /**
   Accumulate data into the buffer, optionally moving
   the write head on by a custom amount.

   This *adds* the content of the frame to whatever is
   already there
   **/
  void push(const_view_type x, size_t frameTime) {
    assert(x.rows() == mChannels);

    size_t blocksize = x.cols();

    assert(blocksize <= bufferSize());

    size_t offset = frameTime;

    if (offset + blocksize > bufferSize()) {
      return;
    }

    offset += mCounter;
    offset = offset < bufferSize() ? offset : offset - bufferSize();

    size_t size = ((offset + blocksize) > bufferSize()) ? bufferSize() - offset
                                                        : blocksize;

    addIn(x(Slice(0), Slice(0, size)), offset, size);
    addIn(x(Slice(0), Slice(size, blocksize - size)), 0, blocksize - size);
  }

  /**
   Copy data from the buffer, and zero where it was
   **/
  void pull(view_type out) {
  
    size_t blocksize = out.cols();
    if (blocksize > bufferSize()) {
      return;
    }

    size_t offset = mCounter;

    size_t size =
        offset + blocksize > bufferSize() ? bufferSize() - offset : blocksize;

    outAndZero(out(Slice(0),Slice(0, size)), offset, size);
    outAndZero(out(Slice(0),Slice(size, blocksize - size)), 0,
               blocksize - size);
  }

//  template <typename OutputIt>
//  void pull(OutputIt out, OutputIt end, size_t nSamps, size_t nChans) {
//    size_t blocksize = nSamps;
//    if (blocksize > bufferSize()) {
//      return;
//    }
//
//    size_t offset = mCounter;
//    size_t size =
//        offset + blocksize > bufferSize() ? bufferSize() - offset : blocksize;
//
//    out_and_zero(out, end, 0, offset, size);
//    out_and_zero(out, end, size, 0, blocksize - size);
//  }

  /*!
   Reset the buffer, resizing if the host buffer size
   or user buffer size have changed.

   This should be called from an audio host's DSP setup routine
   **/
  void reset(size_t channels = 0) {
    if (channels)
      mChannels = channels;

    if (matrix.cols() != bufferSize() || matrix.rows() != channels)
    {
      matrix.resize(mChannels,bufferSize());
      matrix.fill(0);
      mCounter = 0;
    }
  }

  void setSize(size_t n) { mSize = n; }

  void setHostBufferSize(size_t n) { mHostBufferSize = n; }

private:
  void addIn(const_view_type in, size_t offset, size_t size) {
    if (size) {
      matrix(Slice(0), Slice(offset, size)).apply(in, [](double &x, double y) {
        x += y;
      });
    }
  }

  void outAndZero(view_type out, size_t offset, size_t size) {
    if (size) {
      view_type buf = matrix(Slice(0), Slice(offset, size));
      view_type output = buf;
      out = output;
      buf.fill(0);
      mCounter = offset + size;
    }
  }

  template <typename OutputIt>
  void outAndZero(OutputIt out, OutputIt end, size_t outOffset, size_t offset,
                  size_t size) {
    if (size) {
      for (size_t i = 0; (i < mChannels && out != end); ++i, ++out) {
        auto outSlice = matrix(++i, Slice(offset, size));
        *out->copy_to(outSlice, outOffset, size);
      }
      matrix(Slice(0), Slice(offset, size)).fill(0);
      mCounter = offset + size;
    }
  }

  size_t bufferSize() const { return mSize + mHostBufferSize; }

  tensor_type matrix;
  size_t mSize;
  size_t mChannels;
  size_t mCounter = 0;
  size_t mHostBufferSize = 0;
};
} // namespace fluid
