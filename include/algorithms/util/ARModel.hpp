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
#include "../../data/FluidTensor.hpp"
#include "FluidEigenMappings.hpp"
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

  void estimate(FluidTensorView<const double, 1> input, index nIterations = 3,
                double robustFactor = 3.0)
  {
    if (nIterations > 0)
      robustEstimate(input, nIterations, robustFactor);
    else
      directEstimate(input, true);
  }
  
  
  double fowardPrediction(FluidTensorView<const double, 1> input)
  {
    double prediction;
    modelPredict(input, FluidTensorView<double, 1>(&prediction, 0, 1),
                 std::negate<index>{}, Predict{});
    return prediction;
  }

  double backwardPrediction(FluidTensorView<const double, 1> input)
  {
    double prediction;
    modelPredict(input, FluidTensorView<double, 1>(&prediction, 0, 1),
                 Identity{}, Predict{});
    return prediction;
  }

  double forwardError(FluidTensorView<const double, 1> input)
  {
      double error;
      forwardErrorArray(input,FluidTensorView<double,1>(&error,0,1));
      return error;
  }

  double backwardError(FluidTensorView<const double, 1> input)
  {
      double error;
      backwardErrorArray(input,FluidTensorView<double,1>(&error,0,1));
      return error;
  }

  void forwardErrorArray(FluidTensorView<const double, 1> input,FluidTensorView<double, 1> errors)
  {
      modelPredict(input, errors, std::negate<index>{},Error{});
  }

  void backwardErrorArray(FluidTensorView<const double, 1> input, FluidTensorView<double, 1> errors)
  {
      modelPredict(input, errors, Identity{}, Error{});
  }

  void setMinVariance(double variance) { mMinVariance = variance; }

private:
  
  struct Identity
  {
    index operator()(index a) { return a; }
  };
  
  struct Predict
  {
    double operator()(double, double estimate) {return estimate;}
  };
  
  struct Error{
    double operator()(double input, double estimate) {return input -  estimate;}
  };
  
  
  /// \pre Op(numPredictons + mParameters.size()) < input.size() && Op(mParameters.size()) >= -input.descriptor().start
  template <typename Indexer, typename OutputFn>
  void modelPredict(FluidTensorView<const double, 1> input,
                    FluidTensorView<double, 1> output, Indexer f_idx,
                    OutputFn f_out)
  {
    
    index numPredictions = output.size();
    
//    std::cout << ((numPredictions - 1) + f_idx(mParameters.size())) << '\t' << input.size() << '\n';
//    std::cout << f_idx(mParameters.size()) << '\t' << -input.descriptor().start <<  '\n';

    assert(((numPredictions - 1) + f_idx(mParameters.size())) <= input.size() &&
          "array bounds error in AR model prediction: input too short");
    assert(f_idx(mParameters.size()) >= -input.descriptor().start &&
           "array bounds error in AR model prediction: input offset too small");

    const double* input_ptr = input.data();
        
    for(index p = 0; p < numPredictions; p++)
    {
      double estimate = 0;
      for (index i = 0; i < mParameters.size(); i++)
          estimate += mParameters(i) * (input_ptr + p)[f_idx(i + 1)];
      output[p] = f_out(input_ptr[p], estimate);
    }
  }

  void directEstimate(FluidTensorView<const double,1> input, bool updateVariance)
  {
  
    index size = input.size();
  
    // copy input to a 32 byte aligned block (otherwise risk segfaults on Linux)
    VectorXd frame = _impl::asEigen<Eigen::Matrix>(input);
  
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

  void robustEstimate(FluidTensorView<const double, 1> input, index nIterations,
                      double robustFactor)
  {
    FluidTensor<double, 1> estimates(input.size() + mParameters.size());
    
    // Calculate an initial estimate of parameters
    directEstimate(input, true);

    assert(input.descriptor().start >= mParameters.size()&&"too little offset into input data"); 
    
    // Initialise Estimates
    for (index i = 0; i < input.size() + mParameters.size(); i++)
      estimates[i] = input.data()[i - mParameters.size()];

    // Variance
    robustVariance(estimates(Slice(mParameters.size())), input, robustFactor);

    // Iterate
    for (index iterations = nIterations; iterations--;)
      robustIteration(estimates(Slice(mParameters.size())), input,
                      robustFactor);
  }

  double robustResidual(double input, double prediction, double cs)
  {
    return cs * psiFunction((input - prediction) / cs);
  }

  void robustVariance(FluidTensorView<double, 1>       estimates,
                      FluidTensorView<const double, 1> input,
                      double                           robustFactor)
  {
    double residualSqSum = 0.0;

    // Iterate to find new filtered input
    for (index i = 0; i < input.size(); i++)
    {
      const double residual =
          robustResidual(input[i], fowardPrediction(estimates(Slice(i))),
                         robustFactor * sqrt(mVariance));
      residualSqSum += residual * residual;
    }

    setVariance(residualSqSum / input.size());
  }

  void robustIteration(FluidTensorView<double, 1>       estimates,
                       FluidTensorView<const double, 1> input,
                       double                           robustFactor)
  {
    // Iterate to find new filtered input
    for (index i = 0; i < input.size(); i++)
    {
      const double prediction = fowardPrediction(estimates(Slice(i)));
      estimates[i] =
          prediction +
          robustResidual(input[i], prediction, robustFactor * sqrt(mVariance));
    }

    // New parameters
    directEstimate(estimates(Slice(0,input.size())), false);
    robustVariance(estimates(Slice(0,input.size())), input, robustFactor);
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
