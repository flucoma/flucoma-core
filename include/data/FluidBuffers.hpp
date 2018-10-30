/*!
 FluidBuffers.hpp

 Provide input and output buffering

 */
#pragma once

#include "FluidTensor.hpp"
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
  FluidSource() = delete;
  FluidSource(FluidSource &) = delete;
  FluidSource &operator=(FluidSource &) = delete;

  FluidSource(const size_t size, const size_t channels = 1)
      : matrix(channels, size), mSize(size), mChannels(channels) {}

  tensor_type &data() { return matrix; }

  /*
   Push a frame of data into the buffer
   */
  void push(const_view_type x) {
    assert(x.rows() == mChannels);

    size_t blocksize = x.cols();

    assert(blocksize <= bufferSize());

    size_t offset = mCounter;

    size_t size = ((offset + blocksize) > bufferSize()) ? bufferSize() - offset
                                                        : blocksize;

    // Copy all channels (rows)
    copy_in(x(Slice(0), Slice(0, size)), offset, size);
    copy_in(x(Slice(0), Slice(size, blocksize - size)), 0, blocksize - size);
  }

  template <typename U> void push(FluidTensorView<U, 2> x) {
    static_assert(std::is_convertible<U, double>(),
                  "Can't convert between types");

    assert(x.rows() == mChannels);

    size_t blocksize = x.cols();

    assert(blocksize <= bufferSize());

    size_t offset = mCounter;

    size_t size = ((offset + blocksize) > bufferSize()) ? bufferSize() - offset
                                                        : blocksize;

    // Copy all channels (rows)
    copy_in(x(Slice(0), Slice(0, size)), offset, size);
    copy_in(x(Slice(0), Slice(size, blocksize - size)), 0, blocksize - size);
  }

  template <typename InputIt>
  void push(InputIt in, InputIt end, size_t nsamps, size_t nchans) {
    assert(nchans == mChannels);
    assert(nsamps <= bufferSize());
    size_t blocksize = nsamps;

    size_t offset = mCounter;

    size_t size = ((offset + blocksize) > bufferSize()) ? bufferSize() - offset
                                                        : blocksize;

    copyIn(in, end, 0, offset, size);
    copyIn(in, end, size, 0, blocksize - size);
  }

  /*!
   Pull a frame of data out of the buffer.
   */
  void pull(view_type out, size_t frameTime) {
    size_t blocksize = out.cols();
    size_t offset = mHostBufferSize - frameTime;

    if (offset > bufferSize()) {
      out.fill(0);
      return;
    }

    offset += blocksize;
    offset = (offset <= mCounter) ? mCounter - offset
                                  : mCounter + bufferSize() - offset;

    size_t size =
        (offset + blocksize > bufferSize()) ? bufferSize() - offset : blocksize;

    out(Slice(0), Slice(0, size)) = matrix(Slice(0), Slice(offset, size));
    out(Slice(0), Slice(size, blocksize - size)) =
        matrix(Slice(0), Slice(0, blocksize - size));
  }

  /*
   Set the buffer size of the enclosing host.
   Needed to properly handle latency, causality etc
   */
  void setHostBufferSize(const size_t size) { mHostBufferSize = size; }

  /*
   Reset the buffer, resizing if the desired
   size and / or host buffer size have changed

   This should be called in the DSP setup routine of
   the audio host
   */
  void reset(size_t channels = 0) {

    if (channels)
      mChannels = channels;

    if (matrix.cols() != bufferSize() || channels)
      matrix.resize(mChannels, bufferSize());
    matrix.fill(0);
    mCounter = 0;
  }

  void setSize(size_t n) { mSize = n; }

private:
  /*
   Report the size of the whole buffer
   */
  size_t bufferSize() const { return mSize + mHostBufferSize; }

  //    /*
  //     Copy a frame into the buffer and move the write head on
  //     */
  //    void copy_in(const_view_type input, const size_t offset, const size_t
  //    size)
  //    {
  //      if(size)
  //      {
  //        matrix(slice(0),slice(offset,size)) = input;
  //        mCounter = offset + size;
  //      }
  //    }

  template <typename U>
  void copyIn(const FluidTensorView<U, 2> input, const size_t offset,
              const size_t size) {
    if (size) {
      matrix(Slice(0), Slice(offset, size)) = input;
      mCounter = offset + size;
    }
  }

  template <typename InputIt>
  void copyIn(InputIt in, InputIt end, const size_t inStart,
              const size_t offset, const size_t size) {
    if (size) {
      for (size_t i = 0; (i < mChannels && in != end); ++i, ++in) {
        auto inRange = matrix(i, Slice(offset, size));
        (*in)->copyFrom(inRange.row(0), inStart, size);
      }
      mCounter = offset + size;
    }
  }

  tensor_type matrix;
  size_t mCounter = 0;
  size_t mSize;
  size_t mChannels;
  size_t mHostBufferSize = 0;
};

/*
 FluidSink

 An output buffer, with overlap-add
 */
template <typename T> class FluidSink {
  using tensor_type = FluidTensor<T, 2>;
  using view_type = FluidTensorView<T, 2>;
  using const_view_type = const FluidTensorView<T, 2>;

public:
  FluidSink() : FluidSink(65536, 1) {}

  FluidSink(FluidSink &) = delete;
  FluidSink operator=(FluidSink &) = delete;

  FluidSink(const size_t size, const size_t channels = 1)
      : matrix(channels, size), mSize(size), mChannels(channels) {}

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

    outAndZero(out(Slice(0), Slice(0, size)), offset, size);
    outAndZero(out(Slice(0), Slice(size, blocksize - size)), 0,
               blocksize - size);
  }

  template <typename OutputIt>
  void pull(OutputIt out, OutputIt end, size_t nSamps, size_t nChans) {
    size_t blocksize = nSamps;
    if (blocksize > bufferSize()) {
      return;
    }

    size_t offset = mCounter;
    size_t size =
        offset + blocksize > bufferSize() ? bufferSize() - offset : blocksize;

    out_and_zero(out, end, 0, offset, size);
    out_and_zero(out, end, size, 0, blocksize - size);
  }

  /*!
   Reset the buffer, resizing if the host buffer size
   or user buffer size have changed.

   This should be called from an audio host's DSP setup routine
   **/
  void reset(size_t channels = 0) {
    if (channels)
      mChannels = channels;

    if (matrix.cols() != bufferSize() || channels)
      matrix.resize(mChannels, bufferSize());
    matrix.fill(0);
    mCounter = 0;
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
        auto outSlice = matrix(i++, Slice(offset, size));
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
