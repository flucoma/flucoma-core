
#pragma once

#include "ARModel.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace fluid {
namespace transient_extraction {
  
using armodel::ARModel;
using Eigen::MatrixXd;
using Eigen::VectorXd;
    
class TransientExtraction
{
  
public:
  
  TransientExtraction(size_t order, size_t iterations, double robustFactor, double detectHi, double detectLo, int detectHalfWindow, int detectHold, double detectPower) : mModel(order, iterations, robustFactor), mDetectThreshHi(detectLo), mDetectThreshLo(detectHi), mDetectHalfWindow(detectHalfWindow), mDetectHold(detectHold), mDetectPowerFactor(detectPower) {}
  
  size_t size() const { return mSize; }
  
  const double *getDetect() const { return mDetect.data(); }
  const double *getForwardError() const { return mForwardError.data(); }
  const double *getBackwardError() const { return mBackwardError.data(); }
  const double *getForwardWindowedError() const { return mForwardWindowedError.data(); }
  const double *getBackwardWindowedError() const { return mBackwardWindowedError.data(); }
  const double *getCombinedError() const { return mCombinedError.data(); }
  const double *getCombinedWindowedError() const { return mCombinedWindowedError.data(); }

  int detect(const double *input, int size, int pad)
  {
    analyse(input, size, pad);
    return detection(input, size);
  }
  
  int extract(double *output, const double *input, int size, int pad)
  {
    int count = detect(input, size, pad);
    
    if (count)
      interpolate(output, input, mDetect.data(), size, count);
    else
      std::copy(input, input + size, output);
    
    return count;
  }
    
  int extract(double *output, const double *input, const double *unknowns, int size, int pad)
  {
    resizeStorage(size);
   
    int order = mModel.order();
    std::copy(unknowns, unknowns + size, mDetect.data());
    std::fill(mDetect.data(), mDetect.data() + order, 0.0);
    
    int count = 0;
    for (int i = order; i < size; i++)
      if (mDetect[i])
        count++;
    
    if (count)
    {
      analyse(input, size, pad);
      interpolate(output, input, mDetect.data(), size, count);
    }
    else
      std::copy(input, input + size, output);
    
    return count;
  }
  
private:

  void analyse(const double *input, int size, int pad)
  {
    mModel.estimate(input - pad, size + pad + pad);
  }
  
  int detection(const double *input, int size)
  {
    // Resize storage
    
    resizeStorage(size);
    
    // Forward and backward error
    
    const double normFactor = 1.0 / sqrt(mModel.variance());
    
    errorCalculation<&ARModel::forwardErrorArray>(mForwardError.data(), input, size, normFactor);
    errorCalculation<&ARModel::backwardErrorArray>(mBackwardError.data(), input, size, normFactor);
    
    // Window error functions (brute force convolution)
    
    windowError(mForwardWindowedError.data(), mForwardError.data(), size);
    windowError(mBackwardWindowedError.data(), mBackwardError.data(), size);
    
    // Combined error measures
    
    for (int i = 0; i < size; i++)
      mCombinedError[i] = (mForwardError[i] + mBackwardError[i]) * 0.5;
    for (int i = 0; i < size; i++)
      mCombinedWindowedError[i] = (mForwardWindowedError[i] + mBackwardWindowedError[i]) * 0.5;
    
    // Detection
    
    int count = 0;
    const double hiThresh = mDetectThreshHi;
    const double loThresh = mDetectThreshLo;
    const int offHold = mDetectHold;
    
    bool click = false;
    
    for (int i = 0; i < size; i++)
    {
      if (!click && (mBackwardWindowedError[i] > loThresh) && (mForwardWindowedError[i] > hiThresh))
      {
        click = true;
      }
      else if (click && (mBackwardWindowedError[i] < loThresh))
      {
        click = false;
        
        for (int j = i; (j < i + offHold) && (j < size); j++)
        {
          if (mBackwardWindowedError[i] > loThresh)
          {
            click = true;
            break;
          }
        }
      }
      
      if (click)
        count++;
      
      mDetect[i] = click ? 1.0 : 0.0;
    }
    
    return count;      
  }
  
  void interpolate(double *output, const double *input, const double *unknowns, int size, int count)
  {
    const double *parameters = mModel.getParameters();
    int order = mModel.order();
    
    // Declare matrices
    
    MatrixXd A = MatrixXd::Zero(size - order, size);
    MatrixXd U = MatrixXd::Zero(size, count);
    MatrixXd K = MatrixXd::Zero(size, size - count);
    VectorXd xK(size - count);
    
    // Form data
    
    for (int i = 0; i < size - order; i++)
    {
      for (int j = 0; j < order; j++)
        A(i, j + i) = -parameters[order - (j + 1)];
      
      A(i, order + i) = 1.0;
    }
    
    for (int i = 0, uCount = 0, kCount = 0; i < size; i++)
    {
      if (unknowns[i])
        U(i, uCount++) = 1.0;
      else
      {
        K(i, kCount) = 1.0;
        xK[kCount++] = input[i];
      }
    }
    
    // Solve
    
    MatrixXd Au = A * U;
    MatrixXd M = -(Au.transpose() * Au);
    MatrixXd u = M.fullPivLu().solve(Au.transpose() * (A * K) * xK);
    
    // Write the output
    
    for (int i = 0, uCount = 0; i < size; i++)
    {
      if (unknowns[i])
        output[i] = u(uCount++);
      else
        output[i] = input[i];
    }
  }
  
  template <void (ARModel::*Method)(double *, const double *, int)>
  void errorCalculation(double *error, const double *input, int size, double normFactor)
  {
    (mModel.*Method)(error, input, size);

     // Take absolutes and normalise
    
    for (int i = 0; i < size; i++)
      error[i] = std::fabs(error[i]) * normFactor;
  }
  
  // Triangle window
  
  double calcWindow(double norm)
  {
    return std::min(norm, 1.0 - norm);
  }
  
  void windowError(double *errorWindowed, const double *error, int size)
  {
    const int windowSize = mDetectHalfWindow * 2 + 1;
    const int windowOffset = mDetectHalfWindow;
    const double powFactor = mDetectPowerFactor;

    // Calculate window normalisation factor
    
    double windowNormFactor = 0.0;
    
    for (int j = 0; j < windowSize; j++)
    {
      double window = calcWindow((double) j / windowSize);
      windowNormFactor += window;
    }
    
    windowNormFactor = 1.0 / windowNormFactor;
    
    // Do window processing
    
    for (int i = 0; i < size; i++)
    {
      double windowed = 0.0;
      
      for (int j = 1; j < windowSize; j++)
      {
        const double value = pow(fabs(error[i - windowOffset + j]), powFactor);
        windowed += value * calcWindow((double) j / windowSize);;
      }
      
      errorWindowed[i] = pow((windowed * windowNormFactor), 1.0 / powFactor);
    }
  }
  
  void resizeStorage(int size)
  {
    mDetect.resize(size);
    mForwardError.resize(size);
    mBackwardError.resize(size);
    mForwardWindowedError.resize(size);
    mBackwardWindowedError.resize(size);
    mCombinedError.resize(size);
    mCombinedWindowedError.resize(size);
    mSize = size;
  }
  
private:
  
  ARModel mModel;
  
  size_t mSize;

  int mDetectHalfWindow;
  int mDetectHold;

  double mDetectPowerFactor;
  double mDetectThreshHi;
  double mDetectThreshLo;
  
  std::vector<double> mDetect;
  std::vector<double> mForwardError;
  std::vector<double> mBackwardError;
  std::vector<double> mForwardWindowedError;
  std::vector<double> mBackwardWindowedError;
  std::vector<double> mCombinedError;
  std::vector<double> mCombinedWindowedError;
};
  
};  // namespace transient_extraction
};  // namespace fluid

