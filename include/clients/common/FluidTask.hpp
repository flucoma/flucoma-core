#pragma once

#include <atomic>

namespace fluid {

class FluidTask
{
public:
  
  FluidTask() : mProgress(0.0), mCancel(false)
  {}
    
  bool processUpdate(double samplesDone, double taskLength)
  {
    mProgress = (samplesDone / (taskLength * mTotalIterations)) + (mIteration/mTotalIterations);
    return !mCancel;
  }
  
  bool iterationUpdate(double iterationsDone,double totalIterations)
  {
    mIteration = iterationsDone;
    mTotalIterations = totalIterations;
    return !mCancel;
  }
  
  void cancel() { mCancel = true; }
  void reset() { mCancel = false; }
  double progress() { return mProgress; }
  bool cancelled(){ return mCancel; }
    
private:
  std::atomic<double> mProgress;
  bool mCancel;
  double mTotalIterations{1}; //if a wrapped single channel RT process is being run over multiple channels, progress needs reflect the total proportion, rather than going 0->1 n times
  double mIteration{0};
};

} // namespace fluid
