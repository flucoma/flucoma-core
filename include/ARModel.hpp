
#pragma once

#include <cmath>
#include <algorithm>
#include "Eigen/Eigen"
#include "Convolution_Tools.hpp"

namespace fluid {
namespace armodel {
    
class ARModel
{
  
public:
  
  ARModel(size_t order, size_t iterations) : mOrder(order), mIterations(iterations) {}
  
  static Eigen::MatrixXd toeplitz(const Eigen::VectorXd& vec)
  {
    size_t size = vec.size();
    Eigen::MatrixXd mat(size, size);
    
    for (auto i = 0; i < size; i++)
    {
      for (auto j = 0; j < i; j++)
        mat(j, i) = vec(i - j);
      for (auto j = i; j < size; j++)
        mat(j, i) = vec(j - i);
    }
    
    return mat;
  }
  
  Eigen::VectorXd estimate(const double *input, int size, double& variance)
  {
    Eigen::VectorXd autocorrelation(size);
    convolution::autocorrelateReal(autocorrelation.data(), input, size);
    
    // Resize to the desired order (only keep coefficients for up to the order we need)
    
    double pN = mOrder < size ? autocorrelation(mOrder) : autocorrelation(0);
    
    autocorrelation.conservativeResize(mOrder);
    
    // Form a toeplitz matrix
    
    Eigen::MatrixXd mat = toeplitz(autocorrelation);
    
    // Yule Walker
    
    autocorrelation(0) = pN;
    std::rotate(autocorrelation.data(), autocorrelation.data() + 1, autocorrelation.data() + mOrder);
    Eigen::MatrixXd result = mat.llt().solve(autocorrelation);
    
    // Calculate variance
    
    variance = mat(0, 0);
      
    for (int i = 0; i < mOrder - 1; i++)
      variance -= result(i, 0) * mat(0, i + 1);
      
    variance -= result(mOrder - 1, 0) * pN;
    variance /= size;
    
    return result.col(0);
  }
  
  Eigen::VectorXd robustEstimate(const double *input, int size, double& variance)
  {
    // Calculate an intial estimate of parameters
    
    Eigen::VectorXd params = estimate(input, size, variance);
    
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

  double robustFilter(double input, const double *prevEstimates, const double *params, double cs, double& residual)
  {
    double prediction = 0.0;
    
    for (int i = 0; i < mOrder; i++)
      prediction += params[i] * prevEstimates[mOrder - (i + 1)];
   
    residual = (cs * psiFunction(input - prediction)) / cs;
    return prediction + residual;
  }
  
  Eigen::VectorXd robustIteration(double *estimates, const double *input, const double *params, int size, double& deviation)
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

