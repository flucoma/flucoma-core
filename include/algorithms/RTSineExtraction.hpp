#pragma once

#include "algorithms/ConvolutionTools.hpp"
#include "data/FluidEigenMappings.hpp"
#include "data/FluidTensor.hpp"
#include <Eigen/Core>

namespace fluid {
namespace rtsineextraction {

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
using ComplexVector = FluidTensorView<std::complex<double>, 1>;
using ComplexMatrix = FluidTensorView<std::complex<double>, 2>;
using Eigen::Array;
using Eigen::ArrayXcd;
using Eigen::ArrayXd;
using Eigen::ArrayXXcd;
using Eigen::ArrayXXd;
using Eigen::Dynamic;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::RowMajor;
using Eigen::VectorXd;
using fluid::eigenmappings::ArrayXXcdToFluid;
using MatrixXdMap =
    Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using ArrayXXdMap = Map<const Eigen::Array<double, Eigen::Dynamic,
                                           Eigen::Dynamic, Eigen::RowMajor>>;
using ArrayXdMap =
    Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::RowMajor>>;
using ArrayXcdMap = Map<const Array<std::complex<double>, Dynamic, RowMajor>>;

struct SinePeak {
  int centerBin;
  double mag;
  bool assigned;
};

struct SineTrack {
  vector<SinePeak> peaks;
  int startFrame;
  int endFrame;
  bool active;
  bool assigned;
};

class RTSineExtraction {
public:
  RTSineExtraction(int windowSize, int fftSize, int hopSize, int bandwidth,
                   double threshold, int minTrackLength, double magWeight,
                   double freqWeight)
      : mWindowSize(windowSize),
        mWindow(windowFuncs[WindowType::Hann](windowSize)), mFFTSize(fftSize),
        mBins(fftSize / 2 + 1), mFFT(fftSize), mBandwidth(bandwidth),
        mThreshold(threshold), mMinTrackLength(minTrackLength),
        mMagWeight(magWeight), mFreqWeight(freqWeight), mCurrentFrame(0) {
    mWindowTransform = computeWindowTransform(mWindow);
    mOnes = VectorXd::Ones(mBandwidth);
    mWNorm = mWindowTransform.square().sum();
    mBuf = ArrayXXcd::Zero(mBins, minTrackLength);
  }

  void processFrame(const ComplexVector &in, ComplexMatrix out) {
    const auto &epsilon = std::numeric_limits<double>::epsilon;
    mBuf.block(0, 0, mBins, mMinTrackLength - 1) =
        mBuf.block(0, 1, mBins, mMinTrackLength - 1);
    ArrayXcdMap frame(in.data(), mBins);
    mBuf.block(0, mMinTrackLength - 1, mBins, 1) = frame;
    ArrayXd mag = frame.abs().real();
    ArrayXd correlation = getWindowCorrelation(mag);
    vector<int> peaks = findPeaks(correlation, mThreshold);
    peakContinuation(mTracks, peaks, mag);
    vector<SinePeak> sinePeaks = getActivePeaks(mTracks);
    ArrayXd frameSines = additiveSynthesis(sinePeaks);

    ArrayXcd resultFrame = mBuf.col(0);
    ArrayXd resultMag = resultFrame.abs().real();
    ArrayXd frameResidual = resultMag - frameSines;
    frameResidual = (frameResidual < 0).select(0, frameResidual);
    ArrayXXcd result(mBins, 2);

    ArrayXd all = frameSines + frameResidual;
    ArrayXd mult = (1.0 / all.max(epsilon()));
    result.col(0) = resultFrame * (frameSines * mult).min(1.0);
    result.col(1) = resultFrame * (frameResidual * mult).min(1.0);
    out = ArrayXXcdToFluid(result)();
    mCurrentFrame++;
  }

private:
  int mWindowSize;
  vector<double> mWindow;
  int mFFTSize;
  int mBins;
  FFT mFFT;
  VectorXd mOnes;
  ArrayXd mW;
  double mWNorm;
  int mBandwidth;
  double mThreshold;
  ArrayXd mWindowTransform;
  int mMinTrackLength;
  double mMagWeight;
  double mFreqWeight;
  size_t mCurrentFrame;
  vector<SineTrack> mTracks;
  ArrayXXcd mBuf;

  const void peakContinuation(vector<SineTrack> &tracks,
                              const vector<int> peaks, const ArrayXd frame) {
    using std::abs;
    using std::get;
    using std::log;
    using std::sort;
    using std::tuple;
    using std::vector;

    vector<tuple<double, SineTrack *, SinePeak *>> distances;
    for (auto &&track : tracks) {
      track.assigned = false;
    }
    vector<SinePeak> sinePeaks;
    for (auto &&p : peaks) {
      sinePeaks.push_back(SinePeak{p, frame[p]});
    }

    for (auto &track : tracks) {
      if (track.active) {
        for (auto &&peak : sinePeaks) {
          double dist =
              mFreqWeight * abs(log(track.peaks.back().centerBin /
                                    static_cast<double>(peak.centerBin))) +
              mMagWeight * abs(log(track.peaks.back().mag / peak.mag));
          distances.push_back(std::make_tuple(dist, &track, &peak));
        }
      }
    }

    sort(distances.begin(), distances.end(),
         [](tuple<double, SineTrack *, SinePeak *> const &t1,
            tuple<double, SineTrack *, SinePeak *> const &t2) {
           return get<0>(t1) < get<0>(t2);
         });

    for (auto &&pairing : distances) {
      if (!get<1>(pairing)->assigned && !get<2>(pairing)->assigned) {
        get<1>(pairing)->peaks.push_back(*get<2>(pairing));
        get<1>(pairing)->assigned = true;
        get<2>(pairing)->assigned = true;
      }
    }
    // new tracks
    for (auto &&peak : sinePeaks) {
      if (!peak.assigned) {
        tracks.push_back(SineTrack{vector<SinePeak>{peak},
                                   static_cast<int>(mCurrentFrame), -1, true,
                                   true});
      }
    }
    // diying tracks
    for (auto &&track : tracks) {
      if (track.active && !track.assigned) {
        track.active = false;
        track.endFrame = mCurrentFrame;
      }
    }
  }

  vector<SinePeak> getActivePeaks(const vector<SineTrack> tracks) {
    vector<SinePeak> sinePeaks;
    int latencyFrame = mCurrentFrame - mMinTrackLength;
    for (auto &&track : tracks) {
      if (track.startFrame > latencyFrame)
        continue;
      if (track.endFrame >= 0 && track.endFrame <= latencyFrame)
        continue;
      if (track.endFrame >= 0 &&
          track.endFrame - track.startFrame < mMinTrackLength)
        continue;
        
      sinePeaks.push_back(track.peaks[latencyFrame - track.startFrame]);
    }
    return sinePeaks;
  }

  ArrayXd getWindowCorrelation(const ArrayXd frame) {
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
    ArrayXd result = ArrayXd::Zero(mBandwidth);
    ArrayXcd transform = mFFT.process(Map<ArrayXd>(window.data(), mWindowSize));
    for (int i = 0; i < halfBW; i++) {
      result(halfBW + i) = result(halfBW - i) = abs(transform(i));
    }
    return result;
  }

  vector<int> findPeaks(const ArrayXd correlation, double threshold) {
    vector<int> peaks;
    for (int i = 1; i < correlation.size() - 1; i++) {
      if (correlation(i) > correlation(i - 1) &&
          correlation(i) > correlation(i + 1) && correlation(i) > threshold) {
        peaks.push_back(i);
      }
    }
    return peaks;
  }

  ArrayXd additiveSynthesis(const vector<SinePeak> peaks) {
    ArrayXd result = ArrayXd::Zero(mBins);
    for (auto &p : peaks) {
      result += synthesizePeak(p.centerBin, p.mag);
    }
    return result;
  }

  ArrayXd synthesizePeak(int idx, double amp) {
    int halfBW = mBandwidth / 2;
    ArrayXd sine = ArrayXd::Zero(mBins);
    for (int i = idx, j = 0; i < min(idx + halfBW, mBins - 1); i++, j++) {
      sine[i] = amp * mWindowTransform(halfBW + j);
    }
    for (int i = idx, j = 0; i > max(idx - halfBW, 0); i--, j++) {
      sine[i] = amp * mWindowTransform(halfBW - j);
    }
    return sine;
  }
};
} // namespace rtsineextraction
} // namespace fluid
