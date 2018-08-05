
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
  
  TransientExtraction(size_t order, size_t iterations) : mModel(order, iterations) {}
  
  size_t size() const { return mSize; }
  
  const double *getDetect() const { return mDetect.data(); }
  const double *getForwardError() const { return mForwardError.data(); }
  const double *getBackwardError() const { return mBackwardError.data(); }
  const double *getForwardWindowedError() const { return mForwardWindowedError.data(); }
  const double *getBackwardWindowedError() const { return mBackwardWindowedError.data(); }
  const double *getCombinedError() const { return mCombinedError.data(); }
  const double *getCombinedWindowedError() const { return mCombinedWindowedError.data(); }

  int detect(const double *input, const double *analysis, int size, int pad)
  {
    return detection(input, analysis, size, pad);
  }
  
  int extract(double *output, const double *input, const double *analysis, int size, int pad)
  {
    int count = detect(input, analysis, size, pad);
    
    if (count)
      interpolate(output, input, mDetect.data(), size, count);
    else
      std::copy(input, input + size, output);
    
    return count;
  }
  
private:

  int detection(const double *input, const double *analysis, int size, int pad)
  {
    mModel.estimate(analysis - pad, size + pad + pad);
    
    // Resize storage
    
    mDetect.resize(size);
    mForwardError.resize(size);
    mBackwardError.resize(size);
    mForwardWindowedError.resize(size);
    mBackwardWindowedError.resize(size);
    mCombinedError.resize(size);
    mCombinedWindowedError.resize(size);
    
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
    double hiThresh = 3.0;
    double loThresh = 1.5;
    int offHold = 25;
    
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
    
    mSize = size;
    
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
    MatrixXd u = M.llt().solve(Au.transpose() * A * K * xK);
    
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
  
  void windowError(double *errorWindowed, const double *error, int size, int halfWindow = 7)
  {
    int windowSize = halfWindow * 2 + 1;
    int windowOffset = halfWindow;
    
    for (int i = 0; i < size; i++)
    {
      double windowSum = 0.0;
      double windowed = 0.0;
      
      double powFactor = 1.4;
      
      for (int j = 0; j < windowSize; j++)
      {
        double window = calcWindow((double) j / windowSize);
        int pos = i - windowOffset + j;
        
        if (pos >= 0 && pos < size)
        {
          double absValue = fabs(error[pos]);
          double value = pow(absValue, powFactor);;
          windowed += window * value;
          windowSum += window;
        }
      }
      
      errorWindowed[i] = pow((windowed / windowSum), 1.0/powFactor);
    }
  }
  
private:
  
  ARModel mModel;
  
  size_t mSize;

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

