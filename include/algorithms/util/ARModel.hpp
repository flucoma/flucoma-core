/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "FluidEigenMappings.hpp"
#include "FFT.hpp"
#include "Toeplitz.hpp"
#include "../public/WindowFuncs.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/FluidTensor.hpp"
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
  using ArrayXcd = Eigen::ArrayXcd;

public:
  ARModel(index maxOrder, index maxBlockSize, Allocator& alloc)
      : mParameters(maxOrder, alloc),
        mWindow(maxBlockSize + maxOrder, alloc),
        mSpectralIn(nextPower2(2 * maxBlockSize), alloc),
        mSpectralOut(nextPower2(2 * maxBlockSize), alloc),
        mFFT(nextPower2(2 * maxBlockSize), alloc),
        mIFFT(nextPower2(2 * maxBlockSize), alloc)
  {}

  void init(index order) { mParameters.head(order).setZero(); }


  const double* getParameters() const { return mParameters.data(); }
  double        variance() const { return mVariance; }
  index         order() const { return mParameters.size(); }

  void estimate(FluidTensorView<const double, 1> input, index nIterations,
      double robustFactor, Allocator& alloc)
  {
    if (nIterations > 0)
      robustEstimate(input, nIterations, robustFactor, alloc);
    else
      directEstimate(input, true, alloc);
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
    forwardErrorArray(input, FluidTensorView<double, 1>(&error, 0, 1));
    return error;
  }

  double backwardError(FluidTensorView<const double, 1> input)
  {
    double error;
    backwardErrorArray(input, FluidTensorView<double, 1>(&error, 0, 1));
    return error;
  }

  void forwardErrorArray(
      FluidTensorView<const double, 1> input, FluidTensorView<double, 1> errors)
  {
    modelPredict(input, errors, std::negate<index>{}, Error{});
  }

  void backwardErrorArray(
      FluidTensorView<const double, 1> input, FluidTensorView<double, 1> errors)
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
    double operator()(double, double estimate) { return estimate; }
  };

  struct Error
  {
    double operator()(double input, double estimate)
    {
      return input - estimate;
    }
  };

  template <typename Indexer, typename OutputFn>
  void modelPredict(FluidTensorView<const double, 1> input,
      FluidTensorView<double, 1> output, Indexer fIdx, OutputFn fOut)
  {
    index numPredictions = output.size();
    assert(fIdx(1) >= -input.descriptor().start &&
           "array bounds error in AR model prediction: input too short");
    assert(fIdx(1) < input.size() &&
           "array bounds error in AR model prediction: input too short");
    assert(fIdx(mParameters.size()) >= -input.descriptor().start &&
           "array bounds error in AR model prediction: input too short");
    assert(fIdx(mParameters.size()) < input.size() &&
           "array bounds error in AR model prediction: input too short");
    assert(((numPredictions - 1) + fIdx(1)) >= -input.descriptor().start &&
           "array bounds error in AR model prediction: input too short");
    assert(((numPredictions - 1) + fIdx(1)) < input.size() &&
           "array bounds error in AR model prediction: input too short");
    assert(((numPredictions - 1) + fIdx(mParameters.size())) >=
               -input.descriptor().start &&
           "array bounds error in AR model prediction: input too short");
    assert(((numPredictions - 1) + fIdx(mParameters.size())) < input.size() &&
           "array bounds error in AR model prediction: input too short");

    const double* input_ptr = input.data();

    for (index p = 0; p < numPredictions; p++)
    {
      double estimate = 0;
      for (index i = 0; i < mParameters.size(); i++)
        estimate += mParameters(i) * (input_ptr + p)[fIdx(i + 1)];
      output[p] = fOut(input_ptr[p], estimate);
    }
  }

  void directEstimate(FluidTensorView<const double, 1> input,
      bool updateVariance, Allocator& alloc)
  {
    index size = input.size();

    // copy input to a 32 byte aligned block (otherwise risk segfaults on Linux)
    //    FluidEigenMap<const Eigen::Matrix> frame =
    ScopedEigenMap<VectorXd> frame(size, alloc);
    frame = _impl::asEigen<Eigen::Matrix>(input);

    if (mUseWindow)
    {
      if (mWindow.size() != size)
      {
        mWindow.setZero(); // = ArrayXd::Zero(size);
        WindowFuncs::map()[WindowFuncs::WindowTypes::kHann](
            size, mWindow.head(size));
      }

      frame.array() *= mWindow.head(size);
    }

    //    VectorXd autocorrelation(size);
    ScopedEigenMap<VectorXd> autocorrelation(size, alloc);
    autocorrelate(frame, autocorrelation);
    //    algorithm::autocorrelateReal(autocorrelation.data(), frame.data(),

    //                                 asUnsigned(size));

    // Resize to the desired order (only keep coefficients for up to the order
    // we need)
    double pN = mParameters.size() < size ? autocorrelation(mParameters.size())
                                          : autocorrelation(0);
    //    autocorrelation.conservativeResize(mParameters.size());

    // Form a toeplitz matrix
    //    MatrixXd mat = toeplitz(autocorrelation);
    ScopedEigenMap<MatrixXd> mat(mParameters.size(), mParameters.size(), alloc);
    mat = MatrixXd::NullaryExpr(mParameters.size(), mParameters.size(),
        [&autocorrelation, n = mParameters.size()](
            Eigen::Index row, Eigen::Index col) {
          index idx = row - col;
          if (idx < 0) idx += n;
          return autocorrelation(idx);
        });

    // Yule Walker
    autocorrelation(0) = pN;
    std::rotate(autocorrelation.data(), autocorrelation.data() + 1,
        autocorrelation.data() + mParameters.size());
    mParameters = mat.llt().solve(autocorrelation.head(mParameters.size()));

    if (updateVariance)
    {
      // Calculate variance
      double variance = mat(0, 0);

      for (index i = 0; i < mParameters.size() - 1; i++)
        variance -= mParameters(i) * mat(0, i + 1);

      setVariance(
          (variance - (mParameters(mParameters.size() - 1) * pN)) / size);
    }
  }

  void robustEstimate(FluidTensorView<const double, 1> input, index nIterations,
      double robustFactor, Allocator& alloc)
  {
    FluidTensor<double, 1> estimates(input.size() + mParameters.size(), alloc);

    // Calculate an initial estimate of parameters
    directEstimate(input, true, alloc);

    assert(input.descriptor().start >= mParameters.size() &&
           "too little offset into input data");

    // Initialise Estimates
    for (index i = 0; i < input.size() + mParameters.size(); i++)
      estimates[i] = input.data()[i - mParameters.size()];

    // Variance
    robustVariance(estimates(Slice(mParameters.size())), input, robustFactor);

    // Iterate
    for (index iterations = nIterations; iterations--;)
      robustIteration(
          estimates(Slice(mParameters.size())), input, robustFactor, alloc);
  }

  double robustResidual(double input, double prediction, double cs)
  {
    assert(cs > 0); 
    return cs * psiFunction((input - prediction) / cs);
  }

  void robustVariance(FluidTensorView<double, 1> estimates,
      FluidTensorView<const double, 1> input, double robustFactor)
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

  void robustIteration(FluidTensorView<double, 1> estimates,
      FluidTensorView<const double, 1> input, double robustFactor,
      Allocator& alloc)
  {
    // Iterate to find new filtered input
    for (index i = 0; i < input.size(); i++)
    {
      const double prediction = fowardPrediction(estimates(Slice(i)));
      estimates[i] = prediction + robustResidual(input[i], prediction,
                                      robustFactor * sqrt(mVariance));
    }

    // New parameters
    directEstimate(estimates(Slice(0, input.size())), false, alloc);
    robustVariance(estimates(Slice(0, input.size())), input, robustFactor);
  }

  void setVariance(double variance)
  {
    variance = std::max(mMinVariance, variance);
    mVariance = variance;
    assert(mVariance >= 0); 
  }

  // Huber PSI function
  double psiFunction(double x)
  {
    return fabs(x) > 1 ? std::copysign(1.0, x) : x;
  }

  template <typename DerivedA, typename DerivedB>
  void autocorrelate(
      const Eigen::DenseBase<DerivedA>& in, Eigen::DenseBase<DerivedB>& out)
  {
    mSpectralIn.head(in.size()) = in;

    index fftSize = nextPower2(in.size() * 2);
    mFFT.resize(fftSize);
    mIFFT.resize(fftSize);

    mSpectralIn.head(in.size()) = in;

    auto spec = mFFT.process(mSpectralIn.head(in.size()));

    mSpectralOut.head(spec.size()) = spec * spec.conjugate();

    out.head(in.size()) =
        mIFFT.process(mSpectralOut.head(spec.size())).head(in.size());
    out.head(in.size()) /= (fftSize * in.size());
  }

  index nextPower2(index n)
  {
    return static_cast<index>(std::pow(2, std::ceil(std::log2(n))));
  }

  ScopedEigenMap<VectorXd> mParameters;
  double                   mVariance{0.0};
  ScopedEigenMap<ArrayXd>  mWindow;
  ScopedEigenMap<ArrayXd>  mSpectralIn;
  ScopedEigenMap<ArrayXcd> mSpectralOut;
  FFT                      mFFT;
  IFFT                     mIFFT;
  bool                     mUseWindow{true};
  double                   mMinVariance{1e-5};
};

} // namespace algorithm
} // namespace fluid
