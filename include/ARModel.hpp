
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
  
  ARModel(size_t order, size_t iterations) : mOrder(order), mIterations(iterations) {}
  
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
  
  VectorXd estimate(const double *input, int size, double& variance)
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
    MatrixXd result = mat.llt().solve(autocorrelation);
    
    // Calculate variance
    
    variance = mat(0, 0);
      
    for (int i = 0; i < mOrder - 1; i++)
      variance -= result(i, 0) * mat(0, i + 1);
      
    variance -= result(mOrder - 1, 0) * pN;
    variance /= size;
    
    return result.col(0);
  }
  
  VectorXd robustEstimate(const double *input, int size, double& variance)
  {
    // Calculate an intial estimate of parameters
    
    VectorXd params = estimate(input, size, variance);
    
    // Initialise Estimates
    
    std::vector<double> estimates(size + mOrder);
    for (int i = 0; i < mOrder + size; i++)
      estimates[i] = input[i - mOrder];
    
    // Iterate
    
    double deviation = sqrt(variance);
    for (size_t iterations = mIterations; --iterations; )
      params = robustIteration(estimates.data(), input, params.data(), size, deviation);
    
    variance = deviation * deviation;
    
    return params;
  }
  
private:

  template<typename Op>
  double modelEstimate(const double *input, const double *params, int idx)
  {
    double estimate = 0.0;
    
    for (int i = 0; i < mOrder; i++)
      estimate += params[i] * input[Op()(idx, i + 1)];
    
    return estimate;
  }
  
  template<double (ARModel::*Method)(const double *, const double *, int)>
  double modelError(const double *input, const double *params)
  {
      return input[0] - (this->*Method)(input, params, 0);
  }
  
  template<double (ARModel::*Method)(const double *, const double *)>
  void modelErrorArray(double *errors, const double *input, const double *params, int size)
  {
    for (int i = 0; i < size; i++)
      errors[i] = (this->*Method)(input + i, params);
  }

  double fowardEstimate(const double *input, const double *params, int idx)
  {
    return modelEstimate<std::minus<int>>(input, params, idx);
  }
  
  double backwardEstimate(const double *input, const double *params, int idx)
  {
    return modelEstimate<std::plus<int>>(input, params, idx);
  }
  
  double forwardError(const double *input, const double *params)
  {
    return modelError<&ARModel::fowardEstimate>(input, params);
  }
  
  double backwardError(const double *input, const double *params)
  {
    return modelError<&ARModel::backwardEstimate>(input, params);
  }
  
  void forwardErrorArray(double *errors, const double *input, const double *params, int size)
  {
    modelErrorArray<&ARModel::forwardError>(errors, input, params, size);
  }
  
  void backwardErrorArray(double *errors, const double *input, const double *params, int size)
  {
    modelErrorArray<&ARModel::backwardError>(errors, input, params, size);
  }
  
  double robustFilter(double input, const double *prevEstimates, const double *params, double cs, double& residual)
  {
    double prediction = fowardEstimate(prevEstimates, params, mOrder);
    residual = (cs * psiFunction(input - prediction)) / cs;
    return prediction + residual;
  }
  
  VectorXd robustIteration(double *estimates, const double *input, const double *params, int size, double& deviation)
  {
    const double cs = mThreshold * deviation;
    double residualSqSum = 0.0;
    
    // Iterate to find new filtered input
    
    for (int i = 0; i < size; i++)
    {
      double residual;
      
      estimates[i + mOrder] = robustFilter(input[i], estimates + i, params, cs, residual);
      residualSqSum += residual * residual;
    }
    
    // Update deviation
    
    deviation = sqrt(residualSqSum / size);
    
    // New parameters
    
    double variance;
    return estimate(estimates + mOrder, size, variance);
  }
  
  // Huber PSI function
  
  double psiFunction(double x)
  {
    return fabs(x) > 1 ? std::copysign(1.0, x) : x;
  }
  
  size_t mOrder;
  size_t mIterations;
  double mThreshold = 3.0;
};
  
};  // namespace armodel
};  // namespace fluid

