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

#include "../util/ARModel.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace fluid {
namespace algorithm {

class TransientExtraction
{

  using ARModel = algorithm::ARModel;
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;

public:

  TransientExtraction(index maxOrder, index maxBlockSize, index maxPadSize,
      Allocator& alloc = FluidDefaultAllocator())
      : mModel(maxOrder, maxBlockSize + (2 * maxPadSize), alloc),
        mInput(asUnsigned(maxBlockSize + (2 * maxPadSize) + maxOrder), alloc),
        mDetect(asUnsigned(maxBlockSize), alloc),
        mForwardError(asUnsigned(maxBlockSize + maxOrder), alloc),
        mBackwardError(asUnsigned(maxBlockSize + maxOrder), alloc),
        mForwardWindowedError(asUnsigned(maxBlockSize), alloc),
        mBackwardWindowedError(asUnsigned(maxBlockSize), alloc)
  {}

  void init(index order, index blockSize, index padSize)
  {
    mModel.init(order);
    prepareStream(blockSize, padSize);
    mInitialized = true;
  }

  void setDetectionParameters(double power, double threshHi, double threshLo,
      index halfWindow = 7, index hold = 25)
  {
    mDetectPowerFactor = power;
    mDetectThreshHi = threshHi;
    mDetectThreshLo = threshLo;
    mDetectHalfWindow = halfWindow;
    mDetectHold = hold;
  }

  void prepareStream(index blockSize, index padSize)
  {
    mBlockSize = std::max(blockSize, modelOrder());
    mPadSize = std::max(padSize, modelOrder());
    resizeStorage();
  }

  index modelOrder() const { return static_cast<index>(mModel.order()); }
  index blockSize() const { return mBlockSize; }
  index hopSize() const { return mBlockSize - modelOrder(); }
  index padSize() const { return mPadSize; }
  index inputSize() const { return hopSize() + mPadSize; }
  index analysisSize() const { return mBlockSize + mPadSize + mPadSize; }

  const double* getDetect() const { return mDetect.data(); }
  const double* getForwardError() const
  {
    return mForwardError.data() + modelOrder();
  }
  const double* getBackwardError() const
  {
    return mBackwardError.data() + modelOrder();
  }
  const double* getForwardWindowedError() const
  {
    return mForwardWindowedError.data();
  }
  const double* getBackwardWindowedError() const
  {
    return mBackwardWindowedError.data();
  }

  index detect(const double* input, index inSize,
      Allocator& alloc = FluidDefaultAllocator())
  {
    assert(mInitialized && "Call init() before processing");
    assert(modelOrder() >= mDetectHalfWindow &&
           "model order needs to be >= half filter size");
    frame(input, inSize);
    analyze(alloc);
    detection();
    return mCount;
  }

  void process(const RealVectorView input, RealVectorView transients,
      RealVectorView residual, Allocator& alloc = FluidDefaultAllocator())
  {
    assert(mInitialized && "Call init() before processing");
    assert(modelOrder() >= mDetectHalfWindow &&
           "model order needs to be >= half filter size");
    index inSize = input.size();
    frame(input.data(), inSize);
    analyze(alloc);
    detection();
    interpolate(transients.data(), residual.data(), alloc);
  }

  void process(const RealVectorView input, const RealVectorView unknowns,
      RealVectorView transients, RealVectorView residual,
      Allocator& alloc = FluidDefaultAllocator())
  {
    assert(mInitialized && "Call init() before processing");
    assert(modelOrder() >= mDetectHalfWindow &&
           "model order needs to be >= half filter size");
    index inSize = input.size();
    std::copy(unknowns.data(), unknowns.data() + hopSize(), mDetect.data());
    mCount = 0;
    for (index i = 0, size = hopSize(); i < size; i++)
      if (mDetect[asUnsigned(i)] != 0) mCount++;
    frame(input.data(), inSize);
    if (mCount) analyze(alloc);
    interpolate(transients.data(), residual.data(), alloc);
  }

  bool initialized() const { return mInitialized; }

private:
  void frame(const double* input, index inSize)
  {
    using namespace std;
    inSize = std::min(inSize, inputSize());
    copy(mInput.data() + hopSize(),
        mInput.data() + modelOrder() + padSize() + blockSize(), mInput.data());
    copy(input, input + inSize,
        mInput.data() + modelOrder() + padSize() + modelOrder());
    fill(mInput.data() + modelOrder() + padSize() + modelOrder() + inSize,
        mInput.data() + modelOrder() + analysisSize(), 0.0);
  }

  void analyze(Allocator& alloc)
  {
    mModel.setMinVariance(0.0000001);
    mModel.estimate(FluidTensorView<const double, 1>(
                        mInput.data(), modelOrder(), analysisSize()),
        3, 3.0, alloc);
  }

  void detection()
  {
    //    const double* input = mInput.data() + modelOrder() + padSize();

    // Forward and backward error
    const double normFactor = 1.0 / sqrt(mModel.variance());

    auto inputView = FluidTensorView<const double, 1>(mInput.data(),
        modelOrder() + padSize(),
        blockSize() + modelOrder() + mDetectHalfWindow);
    auto fwdError = FluidTensorView<double, 1>(
        mForwardError.data(), 0, blockSize() + mDetectHalfWindow);
    auto backError = FluidTensorView<double, 1>(
        mBackwardError.data(), 0, blockSize() + mDetectHalfWindow);

    errorCalculation<&ARModel::forwardErrorArray>(
        inputView, fwdError, normFactor);
    errorCalculation<&ARModel::backwardErrorArray>(
        inputView, backError, normFactor);

    // Window error functions (brute force convolution)
    auto fwdWindowedError =
        FluidTensorView<double, 1>(mForwardWindowedError.data(), 0, hopSize());
    auto backWindowedError =
        FluidTensorView<double, 1>(mBackwardWindowedError.data(), 0, hopSize());

    windowError(fwdError(Slice(modelOrder())), fwdWindowedError);
    windowError(backError(Slice(modelOrder())), backWindowedError);

    // Detection
    index        count = 0;
    const double hiThresh = mDetectThreshHi;
    const double loThresh = mDetectThreshLo;
    const index  offHold = mDetectHold;

    bool click = false;

    for (index i = 0, size = hopSize(); i < size; i++)
    {
      if (!click && (mBackwardWindowedError[asUnsigned(i)] > loThresh) &&
          (mForwardWindowedError[asUnsigned(i)] > hiThresh))
      {
        click = true;
      }
      else if (click && (mBackwardWindowedError[asUnsigned(i)] < loThresh))
      {
        click = false;

        for (index j = i; (j < i + offHold) && (j < size); j++)
        {
          if (mBackwardWindowedError[asUnsigned(j)] > loThresh)
          {
            click = true;
            break;
          }
        }
      }

      if (click) count++;

      mDetect[asUnsigned(i)] = click ? 1.0 : 0.0;
    }

    // Count Validation
    if (count > (hopSize() / 2))
    {
      std::fill(mDetect.data(), mDetect.data() + hopSize(), 0.0);
      count = 0;
    }

    // RMS validation
    /*
    const double frameRMS = calcStat<&Descriptors::RMS>(input, blockSize());

    for (index i = 0, size = hopSize(); i < size;)
    {
      for (; i < size; i++)
          if (mDetect[i])
            break;

      index beg = i;

      for (; i < size; i++)
        if (!mDetect[i])
          break;

      if (i <= beg)
        continue;

      const double clickRMS = calcStat<&Descriptors::RMS>(input + modelOrder() +
    beg, i - beg);

      if ((clickRMS / frameRMS) < 0.001)
      {
        count -= (i - beg);
        std::fill(mDetect.data() + beg, mDetect.data() + i, 0.0);
      }
    }
    */
    mCount = count;
  }

  template <double Method(const RealVectorView&)>
  double calcStat(const double* input, index size)
  {
    RealVectorView view(const_cast<double*>(input), 0, size);
    return Method(view);
  }

  void interpolate(double* transients, double* residual, Allocator& alloc)
  {
    const double* input = mInput.data() + padSize() + modelOrder();
    const double* parameters = mModel.getParameters();
    index         order = modelOrder();
    index         size = blockSize();

    if (!mCount)
    {
      std::copy(input + order, input + order + hopSize(), residual);
      std::fill_n(transients, hopSize(), 0.0);
      return;
    }

    // Declare matrices
    ScopedEigenMap<MatrixXd> A(size - order, size, alloc);
    ScopedEigenMap<MatrixXd> U(size, mCount, alloc);
    ScopedEigenMap<MatrixXd> K(size, size - mCount, alloc);
    ScopedEigenMap<VectorXd> xK(size - mCount, alloc);

    A.setZero();
    U.setZero();
    K.setZero();
    xK.setZero();

    // Form data
    for (index i = 0; i < size - order; i++)
    {
      for (index j = 0; j < order; j++)
        A(i, j + i) = -parameters[order - (j + 1)];

      A(i, order + i) = 1.0;
    }

    for (index i = 0, uCount = 0, kCount = 0; i < size; i++)
    {
      if (i >= order && mDetect[asUnsigned(i - order)] != 0)
        U(i, uCount++) = 1.0;
      else
      {
        K(i, kCount) = 1.0;
        xK[kCount++] = input[i];
      }
    }

    // Solve
    ScopedEigenMap<MatrixXd> Au(size - order, mCount, alloc);
    Au = A * U;

    ScopedEigenMap<MatrixXd> M(mCount, mCount, alloc);
    M = -(Au.transpose() * Au);

    ScopedEigenMap<VectorXd> u(mCount, alloc);
    u = M.fullPivLu().solve(Au.transpose() * (A * K) * xK);

    // Write the output
    for (index i = 0, uCount = 0; i < (size - order); i++)
    {
      if (mDetect[asUnsigned(i)] != 0)
        residual[i] = u(uCount++);
      else
        residual[i] = input[i + order];
    }

    // if (mRefine) refine(FluidTensorView<double, 1>(residual, 0, size), Au,
    // u);

    for (index i = 0; i < (size - order); i++)
      transients[i] = input[i + order] - residual[i];

    // Copy the residual indexo the correct place
    std::copy(residual, residual + (size - order),
        mInput.data() + padSize() + order + order);
  }

  void refine(
      FluidTensorView<double, 1> io, Eigen::MatrixXd& Au, Eigen::MatrixXd& ls)
  {
    const double energy = mModel.variance() * mCount;
    double       energyLS = 0.0;
    index        order = modelOrder();
    index        size = io.size();

    for (index i = 0; i < (size - order); i++)
    {
      if (mDetect[asUnsigned(i)] != 0)
      {
        const double error = mModel.forwardError(io(Slice(i)));
        energyLS += error * error;
      }
    }

    if (energyLS < energy)
    {
      // Create the square matrix and solve
      Eigen::LLT<Eigen::MatrixXd> M(
          Au.transpose() * Au); // Cholesky decomposition

      Eigen::VectorXd u(mCount);

      Eigen::MatrixXd correction = M.solve(u) + ls;

      // Write the output
      for (index i = 0, uCount = 0; i < (size - order); i++)
      {
        if (mDetect[asUnsigned(i)] != 0) io[i] = u(uCount++);
      }
    }
  }

  double randomSampling(Eigen::VectorXd& output, double variance)
  {
    std::normal_distribution<double> gaussian(0.0, sqrt(variance));
    double                           sum = 0.0;

    for (index i = 0; i < output.size(); i++)
    {
      output[i] = gaussian(mRandomGenerator);
      sum += output[i] * output[i];
    }

    return sum;
  }

  template <void (ARModel::*Method)(
      FluidTensorView<const double, 1>, FluidTensorView<double, 1>)>
  void errorCalculation(FluidTensorView<const double, 1> input,
      FluidTensorView<double, 1> error, double normFactor)
  {
    (mModel.*Method)(input, error);

    // Take absolutes and normalise
    for (index i = 0; i < error.size(); i++)
      error[i] = std::fabs(error[i]) * normFactor;
  }

  // Triangle window
  double calcWindow(double norm) { return std::min(norm, 1.0 - norm); }

  void windowError(FluidTensorView<const double, 1> error,
      FluidTensorView<double, 1>                    errorWindowed)
  {
    assert(error.descriptor().start >= mDetectHalfWindow &&
           "insufficient offset for filter size");
    assert(error.size() > errorWindowed.size() - 1 + (mDetectHalfWindow) &&
           "insufficient input for filter size");

    const index  windowSize = mDetectHalfWindow * 2 + 3;
    const index  windowOffset = mDetectHalfWindow + 1;
    const double powFactor = mDetectPowerFactor;

    // Calculate window normalisation factor
    double windowNormFactor = 0.0;

    for (index j = 0; j < windowSize; j++)
      windowNormFactor += calcWindow((double) j / (windowSize - 1));

    windowNormFactor = 1.0 / windowNormFactor;

    // Do window processing
    for (index i = 0; i < errorWindowed.size(); i++)
    {
      double windowed = 0.0;

      for (index j = 1; j < windowSize - 1; j++)
      {
        const double value = pow(fabs(error[i - windowOffset + j]), powFactor);
        windowed += value * calcWindow((double) j / (windowSize - 1));
      }

      errorWindowed[i] = pow((windowed * windowNormFactor), 1.0 / powFactor);
    }
  }

  void resizeStorage()
  {
    mInput.resize(asUnsigned(analysisSize() + modelOrder()), 0.0);
    mDetect.resize(asUnsigned(hopSize()), 0.0);
    mForwardError.resize(asUnsigned(mBlockSize + modelOrder()), 0.0);
    mBackwardError.resize(asUnsigned(mBlockSize + modelOrder()), 0.0);
    mForwardWindowedError.resize(asUnsigned(hopSize()), 0.0);
    mBackwardWindowedError.resize(asUnsigned(hopSize()), 0.0);
  }

  ARModel mModel;

  std::mt19937_64 mRandomGenerator{std::random_device()()};

  index mBlockSize{0};
  index mPadSize{0};
  index mCount{0};
  //  bool   mRefine{false};
  index  mDetectHalfWindow{1};
  index  mDetectHold{25};
  double mDetectPowerFactor{1.4};
  double mDetectThreshHi{1.5};
  double mDetectThreshLo{3.0};

  rt::vector<double> mInput;
  rt::vector<double> mDetect;
  rt::vector<double> mForwardError;
  rt::vector<double> mBackwardError;
  rt::vector<double> mForwardWindowedError;
  rt::vector<double> mBackwardWindowedError;
  bool               mInitialized{false};
};

} // namespace algorithm
} // namespace fluid
