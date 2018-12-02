/***!
 @file fluid::client::BaseAudioClient

 Provides buffering services, and performs simple pass through (i.e. is a
 concrete class).

 Whilst the buffering classes can do overlap and stuff, we're not set up for
 these here yet.

 Override proccess in derived classes to implement algorithms

 ***/

#pragma once

#include "clients/common/FluidParams.hpp"
#include "data/FluidBuffers.hpp"
#include "data/FluidTensor.hpp"

namespace fluid {
namespace client {

template <typename T, typename U> class BaseAudioClient {
  // Type aliases, mostly for aesthetics
  using source_buffer_type = fluid::FluidSource<T>;
  using sink_buffer_type = fluid::FluidSink<T>;
  using tensor_type = fluid::FluidTensor<T, 2>;
  using view_type = fluid::FluidTensorView<T, 2>;
  using VectorView = fluid::FluidTensorView<T, 1>;
  using const_view_type = const fluid::FluidTensorView<T, 2>;

public:
  template <typename V = U> struct Signal {
  public:
    virtual ~Signal() {}
    virtual void set(V *, V) = 0;
    virtual V &next() = 0;
    virtual void copyFrom(VectorView dst, size_t srcOffset, size_t size) = 0;
    virtual void copyTo(VectorView src, size_t dstOffset, size_t size) = 0;
  };

  class AudioSignal : public Signal<U> {
  public:
    AudioSignal() {}
    AudioSignal(U *ptr, U elem) : mSig(ptr) {}

    void set(U *p, U) override { mSig = p; }

    U &next() override { return *mSig++; }

    void copyFrom(VectorView dst, size_t srcOffset, size_t size) override {
      std::copy(mSig + srcOffset, mSig + srcOffset + size, dst.begin());
    }

    void copyTo(VectorView src, size_t dstOffset, size_t size) override {
      std::copy(src.begin(), src.end(), mSig + dstOffset);
    }

  private:
    U *mSig;
  };

  class ScalarSignal : public Signal<U> {
  public:
    ScalarSignal(){};
    ScalarSignal(U *ptr, U val) : mElem(val) {}

    void set(U *, U p) override { mElem = p; }

    U &next() override { return mElem; }

    virtual void copyFrom(VectorView dst, size_t srcOffset,
                          size_t size) override {
      std::fill(dst.begin(), dst.end(), mElem);
    }

    virtual void copyTo(VectorView src, size_t dstOffset,
                        size_t size) override {
      mElem = *(src.begin());
    }

  private:
    U mElem;
  };

  // No default construction
  BaseAudioClient() = delete;
  // No copying
  BaseAudioClient(BaseAudioClient &) = delete;
  //        BaseAudioClient& operator=(BaseAudioClient&)= delete;
  // Default destuctor
  virtual ~BaseAudioClient() = default;

  /**!
   New instance taking maximum frame size and the number of channels

   You *must* set host buffer size and call reset before attemping to use
   **/
  BaseAudioClient(size_t maxFrameSize, size_t nChannelsIn = 1,
                  size_t nChannelsOut = 1, size_t nIntermediateChannels = 0)
      : mMaxFrameSize(maxFrameSize), mChannelsIn(nChannelsIn),
        mChannelsOut(nChannelsOut),
        mIntermediateChannels(nIntermediateChannels ? nIntermediateChannels
                                                    : nChannelsOut),
        mFrame(0, 0), mFrameOut(0, 0), mFramePost(0, 0),
        mSource(maxFrameSize, nChannelsIn),
        mSink(maxFrameSize, mIntermediateChannels) {
    //          newParamSet();
  }

//  static std::vector<client::Descriptor> &getParamDescriptors() {
//    static std::vector<client::Descriptor> descriptors;
//    if (descriptors.size() == 0) {
//      descriptors.emplace_back("winsize", "Window Size", client::Type::kLong);
//      descriptors.back().setMin(4).setDefault(1024).setInstantiation(true);
//
//      descriptors.emplace_back("hopsize", "Hop Size", client::Type::kLong);
//      descriptors.back().setMin(1).setDefault(512).setInstantiation(true);
//    }
//    return descriptors;
//  }
//
//  static void initParamDescriptors(std::vector<client::Descriptor> &vec) {
//    auto d = getParamDescriptors();
//
//    vec.insert(vec.end(), d.begin(), d.end());
//  }

  /**
   TODO: This works for Max /PD, but wouldn't for SC. Come up with something
   SCish

   - Pushes a buffer from the host into our source buffer
   - Pulls some frames out from the source and processes them
   - Pushes these to the sink buffer
   – Reads back to host buffer

   Don't override this? Maybe? Doesn't seem like a good idea anyway

   Do override process() though. That's what it's for
   **/
  template <typename InputIt, typename OutputIt>
  void doProcess(InputIt input, InputIt inend, OutputIt output, OutputIt outend,
                 size_t nsamps, size_t channelsIn, size_t channelsOut) {
    assert(channelsIn == mChannelsIn);
    assert(channelsOut == mChannelsOut);

    mSource.push(input, inend, nsamps, channelsIn);

    // I had imagined we could delegate knowing about the time into the
    // frame to the buffers, but for cases where chunk_size %
    // host_buffer_size !=0 we don't call the same number of times each tick

    // When we come to worry about overlap, and variable delay times
    // (a) (for overlap) mMaxFrameSize size in this look will need to
    // change to take a variable, from somewhere (representing the hop size
    // for this frame _start_) (b) (for varying frame size) the num rows of
    // the view passed in will need to change.

    for (; mFrameTime < mHostBufferSize; mFrameTime += mHopSize) {
      mSource.pull(mFrame, mFrameTime);
      process(mFrame, mFrameOut);
      if (mChannelsOut)
        mSink.push(mFrameOut, mFrameTime);
    }

    mFrameTime = mFrameTime < mHostBufferSize ? mFrameTime
                                              : mFrameTime - mHostBufferSize;

    if (mChannelsOut) {
      mSink.pull(mFramePost);

      postProcess(mFramePost);

      for (size_t i = 0; (i < mChannelsOut && output != outend);
           ++i, ++output) {
        (*output)->copyTo(mFramePost.row(i), 0, nsamps);
      }
    }
  }

  template <typename InputIt, typename OutputIt>
  void doProcessNoOla(InputIt input, InputIt inend, OutputIt output,
                      OutputIt outend, size_t nsamps, size_t channelsIn,
                      size_t channelsOut) {
    assert(channelsIn == mChannelsIn);
    assert(channelsOut == mChannelsOut);

    mSource.push(input, inend, nsamps, channelsIn);

    // I had imagined we could delegate knowing about the time into the
    // frame to the buffers, but for cases where chunk_size %
    // host_buffer_size !=0 we don't call the same number of times each
    // tick

    // When we come to worry about overlap, and variable delay times
    // (a) (for overlap) mMaxFrameSize size in this look will need to
    // change to take a variable, from somewhere (representing the hop
    // size for this frame _start_) (b) (for varying frame size) the num
    // rows of the view passed in will need to change.

    for (; mFrameTime < mHostBufferSize; mFrameTime += mHopSize) {
      mSource.pull(mFrame, mFrameTime);
      process(mFrame, mFrameOut);

      for (size_t i = 0; (i < mChannelsOut && output != outend);
           ++i, ++output) {
        (*output)->copyTo(mFrameOut.row(i), 0, nsamps);
      }
    }

    mFrameTime = mFrameTime < mHostBufferSize ? mFrameTime
                                              : mFrameTime - mHostBufferSize;
  }

  /**
   Base procesisng method. A no-op in this case
   **/
  virtual void process(view_type in, view_type out) {}
  virtual void postProcess(view_type output) {}

  /**
   Sets the host buffer size. Yes we do need to know this

   Call this from host DSP setup
   **/
  void setHostBufferSize(const size_t size) {
    mHostBufferSize = size;
    mSource.setHostBufferSize(size);
    mSink.setHostBufferSize(size);
    mFramePost.resize(mIntermediateChannels, mHostBufferSize);
  }

  /**
   Reset everything. Call this from host dsp setup
   **/

//  std::tuple<bool, std::string> sanityCheck() {
//    size_t winsize = client::lookupParam("winsize", getParams()).getLong();
//
//    if (winsize > mMaxFrameSize) {
//      return {false, "Window size out of range"};
//    }
//
//    return {true, "All is nice"};
//  }
//
//  virtual void reset(long inputs = -1, long outputs = -1,
//                     long intermediates = -1) {
//
//    mChannelsIn = inputs > -1 ? inputs : mChannelsIn;
//    mChannelsOut = outputs > -1 ? outputs : mChannelsOut;
//    mIntermediateChannels =
//        intermediates > 0 ? intermediates : mIntermediateChannels;
//
//    mFrameTime = 0;
//    mSource.reset(mChannelsIn);
//    mSink.reset(mIntermediateChannels);
//
//    size_t windowSize = getWindowSize();
//    mHopSize = getHopSize();
//
//    if (windowSize != mFrame.cols()) {
//      mFrame = FluidTensor<T, 2>(mChannelsIn, windowSize);
//      mFrameOut = FluidTensor<T, 2>(mIntermediateChannels, windowSize);
//    }
//  }

  size_t channelsOut() { return mChannelsOut; }

  size_t channelsIn() { return mChannelsIn; }

//  virtual size_t getHopSize() {
//    return client::lookupParam("hopsize", getParams()).getLong();
//  }
//
//  virtual size_t getWindowSize() {
//    return client::lookupParam("winsize", getParams()).getLong();
//  }
//
//  virtual std::vector<client::Instance> &getParams() = 0;

private:
  size_t mHostBufferSize;
  size_t mMaxFrameSize;

  size_t mFrameTime;
  size_t mChannelsIn;
  size_t mChannelsOut;
  size_t mIntermediateChannels;

  size_t mHopSize;

  tensor_type mFrame;
  tensor_type mFrameOut;
  tensor_type mFramePost;
  source_buffer_type mSource;
  sink_buffer_type mSink;
};
} // namespace client
} // namespace fluid
