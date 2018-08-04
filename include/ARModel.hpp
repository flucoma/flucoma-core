
#pragma once

#include <cmath>
#include <algorithm>
#include <functional>
#include <Eigen/Eigen>

#include "ConvolutionTools.hpp"
#include "Toeplitz.hpp"
#include "Windows.hpp"

namespace fluid {
namespace armodel {
    
using Eigen::MatrixXd;
using Eigen::VectorXd;
    
class ARModel
{
  
public:
  
  ARModel(size_t order, size_t iterations = 3, bool useWindow = true, double robustFactor = 3.0) : mParameters(VectorXd::Zero(order)), mVariance(0.0), mOrder(order), mIterations(iterations), mUseWindow(useWindow), mRobustFactor(robustFactor)  {}
  
  const double *getParameters() const { return mParameters.data(); }
  double variance() const { return mVariance; }
  size_t order() const { return mOrder; }
  
  void estimate(const double *input, int size)
  {
    if (mIterations)
      robustEstimate(input, size);
    else
      directEstimate(input, size);
  }
  
  double fowardPrediction(const double *input)
  {
    return modelPredict<std::negate<int>>(input);
  }
  
  double backwardPrediction(const double *input)
  {
    struct identity { int operator()(int a) { return a;} };
    return modelPredict<identity>(input);
  }
  
  double forwardError(const double *input)
  {
    return modelError<&ARModel::fowardPrediction>(input);
  }
  
  double backwardError(const double *input)
  {
    return modelError<&ARModel::backwardPrediction>(input);
  }
  
  void forwardErrorArray(double *errors, const double *input, int size)
  {
    modelErrorArray<&ARModel::forwardError>(errors, input, size);
  }
  
  void backwardErrorArray(double *errors, const double *input, int size)
  {
    modelErrorArray<&ARModel::backwardError>(errors, input, size);
  }
  
private:

  template<typename Op>
  double modelPredict(const double *input)
  {
    double estimate = 0.0;
    
    for (int i = 0; i < mOrder; i++)
      estimate += mParameters(i) * input[Op()(i + 1)];
    
    return estimate;
  }
  
  template<double (ARModel::*Method)(const double *)>
  double modelError(const double *input)
  {
      return input[0] - (this->*Method)(input);
  }
  
  template<double (ARModel::*Method)(const double *)>
  void modelErrorArray(double *errors, const double *input, int size)
  {
    for (int i = 0; i < size; i++)
      errors[i] = (this->*Method)(input + i);
  }
  
  void directEstimate(const double *input, int size)
  {
    std::vector<double> frame(size);
    
    if (mUseWindow)
    {
      if (mWindow.size() != size)
      {
        std::vector<double> newWindow = windows::windowFuncs[windows::WindowType::Hann](size);
        std::swap(mWindow, newWindow);
      }
                  
      for (int i = 0; i < size; i++)
        frame[i] = input[i] * mWindow[i] * 2.0;
    }
    else
      std::copy(input, input + size, frame.data());
    
    VectorXd autocorrelation(size);
    convolution::autocorrelateReal(autocorrelation.data(), frame.data(), size);
    
    // Resize to the desired order (only keep coefficients for up to the order we need)
    
    double pN = mOrder < size ? autocorrelation(mOrder) : autocorrelation(0);
    autocorrelation.conservativeResize(mOrder);
    
    // Form a toeplitz matrix
    
    MatrixXd mat = toeplitz(autocorrelation);
    
    // Yule Walker
    
    autocorrelation(0) = pN;
    std::rotate(autocorrelation.data(), autocorrelation.data() + 1, autocorrelation.data() + mOrder);
    mParameters = mat.llt().solve(autocorrelation);
    
    // Calculate variance
    
    double variance = mat(0, 0);
    
    for (int i = 0; i < mOrder - 1; i++)
      variance -= mParameters(i) * mat(0, i + 1);
    
    mVariance = (variance - (mParameters(mOrder - 1) * pN)) / size;
  }
  
  void robustEstimate(const double *input, int size)
  {
    std::vector<double> estimates(size + mOrder);
    
    // Calculate an intial estimate of parameters
    
    directEstimate(input, size);
    
    // Initialise Estimates
    
    for (int i = 0; i < mOrder + size; i++)
      estimates[i] = input[i - mOrder];
    
    // Iterate
    
    for (size_t iterations = mIterations; iterations--; )
      robustIteration(estimates.data() + mOrder, input, size);
  }
  
  double robustFilter(double input, double *estimates, double cs)
  {
    const double prediction = fowardPrediction(estimates);
    const double residual = cs * psiFunction((input - prediction) / cs);
    estimates[0] = prediction + residual;
    return residual * residual;
  }
  
  void robustIteration(double *estimates, const double *input, int size)
  {
    const double cs = mRobustFactor * sqrt(mVariance);
    double residualSqSum = 0.0;
    
    // Iterate to find new filtered input
    
    for (int i = 0; i < size; i++)
      residualSqSum += robustFilter(input[i], estimates + i, cs);
    
    // New parameters
    
    directEstimate(estimates, size);
    
    // Update variance
    
    mVariance = residualSqSum / size;
  }
  
  // Huber PSI function
  
  double psiFunction(double x)
  {
    return fabs(x) > 1 ? std::copysign(1.0, x) : x;
  }
  
  VectorXd mParameters;
  double mVariance;
  
  std::vector<double> mWindow;
  
  bool mUseWindow;
  size_t mOrder;
  size_t mIterations;
  double mRobustFactor;
};
  
};  // namespace armodel
};  // namespace fluid

