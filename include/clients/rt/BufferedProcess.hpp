#pragma once

#include <data/FluidTensor.hpp>
#include <data/FluidBuffers.hpp>

namespace fluid{
namespace client{
class BufferedProcess
{
public:
  template<typename F>
  void process(std::size_t windowSize, std::size_t hopSize, F processFunc){
    
      assert(windowSize <= maxWindowSize() && "Window bigger than maximum");
    
      for( ;mFrameTime< mHostSize; mFrameTime += hopSize)
      {
        if(mSource)
        {
          FluidTensorView<double, 2> windowIn = mFrameIn(Slice(0,windowSize),Slice(0));
          FluidTensorView<double, 2> windowOut = mFrameOut(Slice(0,windowSize),Slice(0));


          mSource->pull(windowIn, mFrameTime);
          processFunc(windowIn, windowOut);
          if (mSink)
            mSink->push(windowOut, mFrameTime);
        }
      }
    
      mFrameTime = mFrameTime < mHostSize ? mFrameTime
                                                : mFrameTime - mHostSize;

    
  }
  
  std::size_t hostSize() const noexcept { return mHostSize; }
  void hostSize(std::size_t size) noexcept { mHostSize = size; }
  
  std::size_t maxWindowSize() const noexcept { return mFrameIn.rows(); }
  void maxWindowSize(std::size_t size) {
    if(size > mFrameIn.rows())
      mFrameIn.resize(size);
    if(size > mFrameOut.rows())
      mFrameOut.resize(size);

  }
  
  void setBuffers(FluidSource<double>& source,FluidSink<double>& sink)
  {
    mSource = &source;
    mSink = &sink; 
  }
  
private:
  std::size_t mFrameTime;
  std::size_t mHostSize;
  FluidTensor<double,2> mFrameIn;
  FluidTensor<double,2> mFrameOut;
  FluidSource<double>* mSource = nullptr;
  FluidSink<double>* mSink = nullptr;
};
} //namespace client
} //namespace fluid
