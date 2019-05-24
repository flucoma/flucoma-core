#pragma once

#include <atomic>

namespace fluid {

class FluidTask
{
public:
  
  FluidTask() : mProgress(0.0), mCancel(false)
  {}
    
  void processUpdate(double samplesDone, double taskLength)
  {
    bool cancel = mCancel;
    mProgress = samplesDone / taskLength;
    mCancel = false;
  }
    
  void cancel() { mCancel = true; }
  void reset() { mCancel = false; }
  double progress() { return mProgress; }
    
private:
  std::atomic<double> mProgress;
  bool mCancel;
};

} // namespace fluid
