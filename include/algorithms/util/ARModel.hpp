/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "ConvolutionTools.hpp"
#include "Toeplitz.hpp"
#include "../public/WindowFuncs.hpp"
#include "../../data/FluidIndex.hpp"
#include <Eigen/Eigen>
#include <algorithm>
#include <cmath>
#include <functional>

namespace fluid {
namespace algorithm {

class ARModel
{

  using MatrixXd = Eigen::MatrixXd;
  using ArrayXd = Eigen::ArrayXd;
  using VectorXd = Eigen::VectorXd;

public:
  ARModel(index order) : mParameters(VectorXd::Zero(order)) {}

  const double* getParameters() const { return mParameters.data(); }
  double        variance() const { return mVariance; }
  index         order() const { return mParameters.size(); }

  void estimate(const double* input, index size, index nIterations = 3,
                double robustFactor = 3.0)
  {
    if (nIterations > 0)
      robustEstimate(input, size, nIterations, robustFactor);
    else
      directEstimate(input, size, true);
  }

  double fowardPrediction(const double* input)
  {
    return modelPredict<std::negate<index>>(input);
  }

  double backwardPrediction(const double* input)
  {
    struct Identity
    {
      index operator()(index a) { return a; }
    };
    return modelPredict<Identity>(input);
  }

  double forwardError(const double* input)
  {
    return modelError<&ARModel::fowardPrediction>(input);
  }

  double backwardError(const double* input)
  {
    return modelError<&ARModel::backwardPrediction>(input);
  }

  void forwardErrorArray(double* errors, const double* input, index size)
  {
    modelErrorArray<&ARModel::forwardError>(errors, input, size);
  }

  void backwardErrorArray(double* errors, const double* input, index size)
  {
    modelErrorArray<&ARModel::backwardError>(errors, input, size);
  }

  void setMinVariance(double variance) { mMinVariance = variance; }

private:
  template <typename Op>
  double modelPredict(const double* input)
  {
    double estimate = 0.0;

    for (index i = 0; i < mParameters.size(); i++)
      estimate += mParameters(i) * input[Op()(i + 1)];

    return estimate;
  }

  template <double (ARModel::*Method)(const double*)>
  double modelError(const double* input)
  {
    return input[0] - (this->*Method)(input);
  }

  template <double (ARModel::*Method)(const double*)>
  void modelErrorArray(double* errors, const double* input, index size)
  {
    for (index i = 0; i < size; i++) errors[i] = (this->*Method)(input + i);
  }

  void directEstimate(const double* input, index size, bool updateVariance)
  {
    // copy input to a 32 byte aligned block (otherwise risk segfaults on Linux)
    VectorXd frame = Eigen::Map<const VectorXd>(input, size);

    if (mUseWindow)
    {
      if (mWindow.size() != size)
      {
        mWindow = ArrayXd::Zero(size);
        WindowFuncs::map()[WindowFuncs::WindowTypes::kHann](size, mWindow);
      }

      frame.array() *= mWindow;
    }


    VectorXd autocorrelation(size);
    algorithm::autocorrelateReal(autocorrelation.data(), frame.data(),
                                 asUnsigned(size));

    // Resize to the desired order (only keep coefficients for up to the order
    // we need)
    double pN = mParameters.size() < size ? autocorrelation(mParameters.size())
                                          : autocorrelation(0);
    autocorrelation.conservativeResize(mParameters.size());

    // Form a toeplitz matrix
    MatrixXd mat = toeplitz(autocorrelation);

    // Yule Walker
    autocorrelation(0) = pN;
    std::rotate(autocorrelation.data(), autocorrelation.data() + 1,
                autocorrelation.data() + mParameters.size());
    mParameters = mat.llt().solve(autocorrelation);

    if (updateVariance)
    {
      // Calculate variance
      double variance = mat(0, 0);

      for (index i = 0; i < mParameters.size() - 1; i++)
        variance -= mParameters(i) * mat(0, i + 1);

      setVariance((variance - (mParameters(mParameters.size() - 1) * pN)) /
                  size);
    }
  }

  void robustEstimate(const double* input, index size, index nIterations,
                      double robustFactor)
  {
    std::vector<double> estimates(asUnsigned(size + mParameters.size()));

    // Calculate an initial estimate of parameters
    directEstimate(input, size, true);

    // Initialise Estimates
    for (index i = 0; i < mParameters.size() + size; i++)
      estimates[asUnsigned(i)] = input[i - mParameters.size()];

    // Variance
    robustVariance(estimates.data() + mParameters.size(), input, size,
                   robustFactor);

    // Iterate
    for (index iterations = nIterations; iterations--;)
      robustIteration(estimates.data() + mParameters.size(), input, size,
                      robustFactor);
  }

  double robustResidual(double input, double prediction, double cs)
  {
    return cs * psiFunction((input - prediction) / cs);
  }

  void robustVariance(double* estimates, const double* input, index size,
                      double robustFactor)
  {
    double residualSqSum = 0.0;

    // Iterate to find new filtered input
    for (index i = 0; i < size; i++)
    {
      const double residual =
          robustResidual(input[i], fowardPrediction(estimates + i),
                         robustFactor * sqrt(mVariance));
      residualSqSum += residual * residual;
    }

    setVariance(residualSqSum / size);
  }

  void robustIteration(double* estimates, const double* input, index size,
                       double robustFactor)
  {
    // Iterate to find new filtered input
    for (index i = 0; i < size; i++)
    {
      const double prediction = fowardPrediction(estimates + i);
      estimates[i] =
          prediction +
          robustResidual(input[i], prediction, robustFactor * sqrt(mVariance));
    }

    // New parameters
    directEstimate(estimates, size, false);
    robustVariance(estimates, input, size, robustFactor);
  }

  void setVariance(double variance)
  {
    if (variance > 0) variance = std::max(mMinVariance, variance);
    mVariance = variance;
  }

  // Huber PSI function
  double psiFunction(double x)
  {
    return fabs(x) > 1 ? std::copysign(1.0, x) : x;
  }

  VectorXd mParameters;
  double   mVariance{0.0};
  ArrayXd  mWindow;
  bool     mUseWindow{true};
  double   mMinVariance{0.0};
};

} // namespace algorithm
} // namespace fluid
