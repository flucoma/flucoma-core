/*!
 FluidBuffers.hpp

 Provide input and output buffering

 */
#pragma once

#include <cassert>
#include <data/FluidTensor.hpp>

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
  FluidSource(const FluidSource &) = delete;
  FluidSource &operator=(const FluidSource &) = delete;
  FluidSource(FluidSource&&) = default;
  FluidSource& operator=(FluidSource&&) = default; 

  FluidSource(const size_t size, const size_t channels = 1)
      : matrix(channels, size), mSize(size), mChannels(channels) {}

  FluidSource() : FluidSource(0, 1){};

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

    // Copy all channels (cols)
    copyIn(x(Slice(0), Slice(0, size)), offset, size);
    copyIn(x(Slice(0), Slice(size, blocksize - size)), 0, blocksize - size);
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
    copyIn(x(Slice(0), Slice(0, size)), offset, size);
    copyIn(x(Slice(0), Slice(size, blocksize - size)), 0, blocksize - size);
  }

  //  template <typename InputIt>
  //  void push(InputIt in, InputIt end, size_t nsamps, size_t nchans) {
  //    assert(nchans == mChannels);
  //    assert(nsamps <= bufferSize());
  //    size_t blocksize = nsamps;
  //
  //    size_t offset = mCounter;
  //
  //    size_t size = ((offset + blocksize) > bufferSize()) ? bufferSize() -
  //    offset
  //                                                        : blocksize;
  //
  //    copyIn(in, end, 0, offset, size);
  //    copyIn(in, end, size, 0, blocksize - size);
  //  }

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

    if (matrix.cols() != bufferSize() || matrix.rows() != channels) {
      matrix.resize(mChannels, bufferSize());
      matrix.fill(0);
      mCounter = 0;
    }
  }

  void setSize(size_t n) { mSize = n; }

  size_t channels() const noexcept { return mChannels; }
  size_t size() const noexcept { return mSize; }
  size_t hostBufferSize() const noexcept { return mHostBufferSize; }

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
} // namespace fluid
