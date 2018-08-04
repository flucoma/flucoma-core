
#pragma once

#include <cmath>
#include <algorithm>
#include <functional>
#include <Eigen/Eigen>
#include "ConvolutionTools.hpp"

namespace fluid {
namespace armodel {
    
using Eigen::MatrixXd;
using Eigen::VectorXd;
    
class ARModel
{
  
public:
  
  ARModel(size_t order, size_t iterations) : mParameters(VectorXd::Zero(order)), mVariance(0.0), mOrder(order), mIterations(iterations) {}
  
  const double *getParameters() const { return mParameters.data(); }
  double variance() const { return mVariance; }
  size_t order() const { return mOrder; }
  
  static MatrixXd toeplitz(const VectorXd& vec)
  {
    size_t size = vec.size();
    MatrixXd mat(size, size);
    
    for (auto i = 0; i < size; i++)
    {
      for (auto j = 0; j < i; j++)
        mat(j, i) = vec(i - j);
      for (auto j = i; j < size; j++)
        mat(j, i) = vec(j - i);
    }
    
    return mat;
  }
  
  void estimate(const double *input, int size)
  {
    VectorXd autocorrelation(size);
    convolution::autocorrelateReal(autocorrelation.data(), input, size);
    
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
      
    variance -= mParameters(mOrder - 1) * pN;
    mVariance = variance / size;
  }
  
  void robustEstimate(const double *input, int size)
  {
    // Calculate an intial estimate of parameters
    
    estimate(input, size);
    
    // Initialise Estimates
    
    std::vector<double> estimates(size + mOrder);
    for (int i = 0; i < mOrder + size; i++)
      estimates[i] = input[i - mOrder];
    
    // Iterate
    
    for (size_t iterations = mIterations; --iterations; )
      robustIteration(estimates.data(), input, size);
  }
  
  double fowardPrediction(const double *input, int idx)
  {
    return modelPredict<std::minus<int>>(input, idx);
  }
  
  double backwardPrediction(const double *input, int idx)
  {
    return modelPredict<std::plus<int>>(input, idx);
  }
  
  double forwardError(const double *input)
  {
    return modelError<&ARModel::fowardPrediction>(input, 0);
  }
  
  double backwardError(const double *input)
  {
    return modelError<&ARModel::backwardPrediction>(input, 0);
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
  double modelPredict(const double *input, int idx)
  {
    double estimate = 0.0;
    
    for (int i = 0; i < mOrder; i++)
      estimate += mParameters(i) * input[Op()(idx, i + 1)];
    
    return estimate;
  }
  
  template<double (ARModel::*Method)(const double *, int)>
  double modelError(const double *input, const double *params)
  {
      return input[0] - (this->*Method)(input, 0);
  }
  
  template<double (ARModel::*Method)(const double *)>
  void modelErrorArray(double *errors, const double *input, int size)
  {
    for (int i = 0; i < size; i++)
      errors[i] = (this->*Method)(input + i);
  }
  
  double robustFilter(double input, const double *prevEstimates, double cs, double& residual)
  {
    double prediction = fowardPrediction(prevEstimates, mOrder);
    residual = (cs * psiFunction(input - prediction)) / cs;
    return prediction + residual;
  }
  
  void robustIteration(double *estimates, const double *input, int size)
  {
    const double cs = mThreshold * sqrt(mVariance);
    double residualSqSum = 0.0;
    
    // Iterate to find new filtered input
    
    for (int i = 0; i < size; i++)
    {
      double residual;
      
      estimates[i + mOrder] = robustFilter(input[i], estimates + i, cs, residual);
      residualSqSum += residual * residual;
    }
    
    // New parameters
    
    estimate(estimates + mOrder, size);
    
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
  
  size_t mOrder;
  size_t mIterations;
  double mThreshold = 3.0;
};
  
};  // namespace armodel
};  // namespace fluid

