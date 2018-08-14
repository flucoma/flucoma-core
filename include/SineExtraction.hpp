#pragma once

#include <ConvolutionTools.hpp>
#include <Eigen/Core>
#include <FluidTensor.hpp>

namespace fluid {
namespace sineextraction {

using convolution::correlateReal;
using convolution::kEdgeWrapCentre;
using fft::FFT;
using std::abs;
using std::complex;
using std::max;
using std::min;
using std::vector;
using windows::windowFuncs;
using windows::WindowType;

using RealMatrix = FluidTensor<double, 2>;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using MatrixXdMap =
    Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using ArrayXXdMap =
    Map<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using ArrayXdMap = Map<Eigen::Array<double, Eigen::Dynamic, Eigen::RowMajor>>;

struct SinesPlusNoiseModel {
  RealMatrix sines;
  RealMatrix noise;
};

class SineExtraction {
public:
  SineExtraction(int windowSize, int fftSize, int hopSize, int bandwidth,
            double threshold)
      : mWindowSize(windowSize),
        mWindow(windowFuncs[WindowType::Hann](windowSize)), mFFTSize(fftSize),
        mFFT(fftSize), mBandwidth(bandwidth), mThreshold(threshold) {
    mWindowTransform = computeWindowTransform(mWindow);
    mOnes = VectorXd::Ones(mBandwidth);
    mWNorm = mWindowTransform.square().sum();
  }

  const SinesPlusNoiseModel process(const RealMatrix &X) {
    ArrayXXdMap input(X.data(), X.extent(0), X.extent(1));
    SinesPlusNoiseModel result;
    result.sines = RealMatrix(X.rows(), X.cols());
    result.noise = RealMatrix(X.rows(), X.cols());
    MatrixXd sines(X.rows(), X.cols());
    MatrixXd noise(X.rows(), X.cols());

    for (int i = 0; i < X.rows(); i++) {
      ArrayXdMap frame(X.row(i).data(), X.cols());
      ArrayXd correlation = getWindowCorrelation(frame);
      vector<int> peaks = findPeaks(correlation, mThreshold);
      ArrayXd frameSines = additiveSynthesis(peaks, frame);
      ArrayXd frameResidual = frame - frameSines;
      frameResidual = (frameResidual < 0).select(0, frameResidual);
      sines.row(i) = correlation;
      noise.row(i) = frameResidual;
    }
    MatrixXdMap(result.sines.data(), X.rows(), X.cols()) = sines;
    MatrixXdMap(result.noise.data(), X.rows(), X.cols()) = noise;

    return result;
  }

private:
  int mWindowSize;
  vector<double> mWindow;
  int mFFTSize;
  FFT mFFT;
  VectorXd mOnes;
  ArrayXd mW;
  double mWNorm;
  int mBandwidth;
  double mThreshold;
  ArrayXd mWindowTransform;

  ArrayXd getWindowCorrelation(ArrayXd frame) {
    ArrayXd squareMag = frame.square();
    ArrayXd corr(frame.size());
    ArrayXd spectrumNorm(frame.size());
    correlateReal(corr.data(), frame.data(), frame.size(),
                  mWindowTransform.data(), mBandwidth, kEdgeWrapCentre);
    convolveReal(spectrumNorm.data(), squareMag.data(), frame.size(),
                 mOnes.data(), mBandwidth, kEdgeWrapCentre);
    corr = corr.square() / (spectrumNorm * mWNorm);
    return corr;
  }

  ArrayXd computeWindowTransform(vector<double> window) {
    int halfBW = mBandwidth / 2;
    vector<double> result(mBandwidth, 0);
    vector<complex<double>> transform = mFFT.process(window);
    for (int i = 0; i <= halfBW; i++) {
      result[halfBW + i] = result[halfBW - i] = abs(transform[i]);
    }
    return ArrayXdMap(result.data(), mBandwidth);
  }

  vector<int> findPeaks(ArrayXd correlation, double threshold) {
    vector<int> peaks;
    for (int i = 1; i < correlation.size() - 1; i++) {
      if (correlation(i) > correlation(i - 1) &&
          correlation(i) > correlation(i + 1) && correlation(i) > threshold) {
        peaks.push_back(i);
      }
    }
    return peaks;
  }

  ArrayXd additiveSynthesis(vector<int> peaks, ArrayXd spectrum) {
    ArrayXd result = ArrayXd::Zero(mFFTSize / 2 + 1);
    for (auto &p : peaks) {
      result += synthesizePeak(p, spectrum(p));
    }
    return result;
  }

  ArrayXd synthesizePeak(int idx, double amp) {
    int halfBW = mBandwidth / 2;
    ArrayXd sine = ArrayXd::Zero(mFFTSize / 2 + 1);
    int frameSize = mFFTSize / 2 + 1;
    for (int i = idx, j = 0; i < min(idx + halfBW, frameSize - 1); i++, j++) {
      sine[i] = amp * mWindowTransform(halfBW + j);
    }
    for (int i = idx, j = 0; i > max(idx - halfBW, 0); i--, j++) {
      sine[i] = amp * mWindowTransform(halfBW - j);
    }
    return sine;
  }
};
} // namespace sineextraction
} // namespace fluid
