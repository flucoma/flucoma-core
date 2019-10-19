/*
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See LICENSE file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
#pragma once

#include "../../data/TensorTypes.hpp"
#include "../util/ARModel.hpp"
#include "../util/FluidEigenMappings.hpp"

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
  TransientExtraction(size_t order, size_t iterations, double robustFactor,
                      bool refine)
      : mModel(order, iterations, robustFactor)
  {}

  void init(size_t order, size_t iterations, double robustFactor, bool refine,
            int blockSize, int padSize)
  {
    mModel = ARModel(order, iterations, robustFactor);
    prepareStream(blockSize, padSize);
    mInitialized = true;
  }

  void setDetectionParameters(double power, double threshHi, double threshLo,
                              int halfWindow = 7, int hold = 25)
  {
    mDetectPowerFactor = power;
    mDetectThreshHi = threshHi;
    mDetectThreshLo = threshLo;
    mDetectHalfWindow = halfWindow;
    mDetectHold = hold;
  }

  void prepareStream(int blockSize, int padSize)
  {
    mBlockSize = std::max(blockSize, modelOrder());
    mPadSize = std::max(padSize, modelOrder());
    resizeStorage();
  }

  int modelOrder() const { return static_cast<int>(mModel.order()); }
  int blockSize() const { return mBlockSize; }
  int hopSize() const { return mBlockSize - modelOrder(); }
  int padSize() const { return mPadSize; }
  int inputSize() const { return hopSize() + mPadSize; }
  int analysisSize() const { return mBlockSize + mPadSize + mPadSize; }

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

  int detect(const double* input, int inSize)
  {
    frame(input, inSize);
    analyse();
    detection();

    return mCount;
  }

  // int extract(double *transients, double *residual, const double *input,
  //            int inSize) {

  void process(const RealVectorView input, RealVectorView transients,
               RealVectorView residual)
  {
    int inSize = input.extent(0);
    frame(input.data(), inSize);
    analyse();
    detection();
    interpolate(transients.data(), residual.data());
    // return mCount;
  }

  // int extract(double *transients, double *residual, const double *input,
  //            int inSize, const double *unknowns) {
  void process(const RealVectorView input, const RealVectorView unknowns,
               RealVectorView transients, RealVectorView residual)
  {
    int inSize = input.extent(0);
    std::copy(unknowns.data(), unknowns.data() + hopSize(), mDetect.data());
    mCount = 0;
    for (int i = 0, size = hopSize(); i < size; i++)
      if (mDetect[i]) mCount++;
    frame(input.data(), inSize);
    if (mCount) analyse();
    interpolate(transients.data(), residual.data());
    // return mCount;
  }

  bool initialized() { return mInitialized; }

private:
  void frame(const double* input, int inSize)
  {
    inSize = std::min(inSize, inputSize());
    std::copy(mInput.data() + hopSize(),
              mInput.data() + modelOrder() + padSize() + blockSize(),
              mInput.data());
    std::copy(input, input + inSize,
              mInput.data() + modelOrder() + padSize() + modelOrder());
    std::fill(mInput.data() + modelOrder() + padSize() + modelOrder() + inSize,
              mInput.data() + modelOrder() + analysisSize(), 0.0);
  }

  void analyse()
  {
    mModel.setMinVariance(0.0000001);
    mModel.estimate(mInput.data() + modelOrder(), analysisSize());
  }

  void detection()
  {
    const double* input = mInput.data() + modelOrder() + padSize();

    // Forward and backward error

    const double normFactor = 1.0 / sqrt(mModel.variance());

    errorCalculation<&ARModel::forwardErrorArray>(
        mForwardError.data(), input, blockSize() + mDetectHalfWindow + 1,
        normFactor);
    errorCalculation<&ARModel::backwardErrorArray>(
        mBackwardError.data(), input, blockSize() + mDetectHalfWindow + 1,
        normFactor);

    // Window error functions (brute force convolution)

    windowError(mForwardWindowedError.data(),
                mForwardError.data() + modelOrder(), hopSize());
    windowError(mBackwardWindowedError.data(),
                mBackwardError.data() + modelOrder(), hopSize());

    // Detection

    int          count = 0;
    const double hiThresh = mDetectThreshHi;
    const double loThresh = mDetectThreshLo;
    const int    offHold = mDetectHold;

    bool click = false;

    for (int i = 0, size = hopSize(); i < size; i++)
    {
      if (!click && (mBackwardWindowedError[i] > loThresh) &&
          (mForwardWindowedError[i] > hiThresh))
      {
        click = true;
      } else if (click && (mBackwardWindowedError[i] < loThresh))
      {
        click = false;

        for (int j = i; (j < i + offHold) && (j < size); j++)
        {
          if (mBackwardWindowedError[j] > loThresh)
          {
            click = true;
            break;
          }
        }
      }

      if (click) count++;

      mDetect[i] = click ? 1.0 : 0.0;
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

    for (int i = 0, size = hopSize(); i < size;)
    {
      for (; i < size; i++)
          if (mDetect[i])
            break;

      int beg = i;

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
  double calcStat(const double* input, int size)
  {
    RealVectorView view(const_cast<double*>(input), 0, size);
    return Method(view);
  }

  void interpolate(double* transients, double* residual)
  {
    const double* input = mInput.data() + padSize() + modelOrder();
    const double* parameters = mModel.getParameters();
    int           order = modelOrder();
    int           size = blockSize();

    if (!mCount)
    {
      std::copy(input + order, input + order + hopSize(), residual);
      std::fill_n(transients, hopSize(), 0.0);
      return;
    }

    // Declare matrices

    MatrixXd A = MatrixXd::Zero(size - order, size);
    MatrixXd U = MatrixXd::Zero(size, mCount);
    MatrixXd K = MatrixXd::Zero(size, size - mCount);
    VectorXd xK(size - mCount);

    // Form data

    for (int i = 0; i < size - order; i++)
    {
      for (int j = 0; j < order; j++)
        A(i, j + i) = -parameters[order - (j + 1)];

      A(i, order + i) = 1.0;
    }

    for (int i = 0, uCount = 0, kCount = 0; i < size; i++)
    {
      if (i >= order && mDetect[i - order])
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

    for (int i = 0, uCount = 0; i < (size - order); i++)
    {
      if (mDetect[i])
        residual[i] = u(uCount++);
      else
        residual[i] = input[i + order];
    }

    if (mRefine) refine(residual, size, Au, u);

    for (int i = 0; i < (size - order); i++)
      transients[i] = input[i + order] - residual[i];

    // Copy the residual into the correct place

    std::copy(residual, residual + (size - order),
              mInput.data() + padSize() + order + order);
  }

  void refine(double* io, int size, Eigen::MatrixXd& Au, Eigen::MatrixXd& ls)
  {
    const double energy = mModel.variance() * mCount;
    double       energyLS = 0.0;
    int          order = modelOrder();

    for (int i = 0; i < (size - order); i++)
    {
      if (mDetect[i])
      {
        const double error = mModel.forwardError(io + i);
        energyLS += error * error;
      }
    }

    if (energyLS < energy)
    {
      // Create the square matrix and solve

      Eigen::LLT<Eigen::MatrixXd> M(Au.transpose() *
                                    Au); // Cholesky decomposition
      Eigen::VectorXd             u(mCount);

      double          sum = randomSampling(u, (energy - energyLS) / mCount);
      Eigen::MatrixXd correction = M.solve(u) + ls;

      // Write the output

      for (int i = 0, uCount = 0; i < (size - order); i++)
      {
        if (mDetect[i]) io[i] = u(uCount++);
      }

      std::cout << "Energy is " << energyLS << " expected " << energy << "\n";
      std::cout << "Energy is " << sum << " should be " << (energy - energyLS)
                << "\n";
      if (energyLS > sum) std::cout << "******ENERGY DECREASE******\n";
    }
  }

  double randomSampling(Eigen::VectorXd& output, double variance)
  {
    std::normal_distribution<double> gaussian(0.0, sqrt(variance));
    double                           sum = 0.0;

    for (int i = 0; i < output.size(); i++)
    {
      output[i] = gaussian(mRandomGenerator);
      sum += output[i] * output[i];
    }

    return sum;
  }

  template <void (ARModel::*Method)(double*, const double*, int)>
  void errorCalculation(double* error, const double* input, int size,
                        double normFactor)
  {
    (mModel.*Method)(error, input, size);

    // Take absolutes and normalise

    for (int i = 0; i < size; i++) error[i] = std::fabs(error[i]) * normFactor;
  }

  // Triangle window

  double calcWindow(double norm) { return std::min(norm, 1.0 - norm); }

  void windowError(double* errorWindowed, const double* error, int size)
  {
    const int    windowSize = mDetectHalfWindow * 2 + 1;
    const int    windowOffset = mDetectHalfWindow;
    const double powFactor = mDetectPowerFactor;

    // Calculate window normalisation factor

    double windowNormFactor = 0.0;

    for (int j = 0; j < windowSize; j++)
      windowNormFactor += calcWindow((double) j / windowSize);

    windowNormFactor = 1.0 / windowNormFactor;

    // Do window processing

    for (int i = 0; i < size; i++)
    {
      double windowed = 0.0;

      for (int j = 1; j < windowSize; j++)
      {
        const double value = pow(fabs(error[i - windowOffset + j]), powFactor);
        windowed += value * calcWindow((double) j / windowSize);
        ;
      }

      errorWindowed[i] = pow((windowed * windowNormFactor), 1.0 / powFactor);
    }
  }

  void resizeStorage()
  {
    mInput.resize(analysisSize() + modelOrder(), 0.0);
    mDetect.resize(hopSize(), 0.0);
    mForwardError.resize(mBlockSize + modelOrder(), 0.0);
    mBackwardError.resize(mBlockSize + modelOrder(), 0.0);
    mForwardWindowedError.resize(hopSize(), 0.0);
    mBackwardWindowedError.resize(hopSize(), 0.0);
  }

  ARModel mModel;

  std::mt19937_64 mRandomGenerator{std::random_device()()};

  int    mBlockSize{0};
  int    mPadSize{0};
  int    mCount{0};
  bool   mRefine{false};
  int    mDetectHalfWindow{1};
  int    mDetectHold{25};
  double mDetectPowerFactor{1.4};
  double mDetectThreshHi{1.5};
  double mDetectThreshLo{3.0};

  std::vector<double> mInput;
  std::vector<double> mDetect;
  std::vector<double> mForwardError;
  std::vector<double> mBackwardError;
  std::vector<double> mForwardWindowedError;
  std::vector<double> mBackwardWindowedError;
  bool                mInitialized{false};
};

}; // namespace algorithm
}; // namespace fluid
