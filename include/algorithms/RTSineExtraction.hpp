#pragma once


#include "algorithms/FFT.hpp"
#include "algorithms/Windows.hpp"
#include "algorithms/ConvolutionTools.hpp"
#include "data/FluidEigenMappings.hpp"
#include "data/FluidTensor.hpp"
#include <Eigen/Core>
#include <queue>

namespace fluid {
namespace rtsineextraction {

using convolution::correlateReal;
using convolution::kEdgeWrapCentre;
using fft::FFT;
using std::vector;
using windows::windowFuncs;
using windows::WindowType;

using Eigen::Array;
using Eigen::ArrayXcd;
using Eigen::ArrayXd;
using Eigen::Dynamic;
using Eigen::Map;
using Eigen::RowMajor;
using Eigen::VectorXd;


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
  using RealMatrix = FluidTensor<double, 2>;
  using ComplexVector = FluidTensorView<std::complex<double>, 1>;
  using ComplexMatrix = FluidTensorView<std::complex<double>, 2>;

  RTSineExtraction(int windowSize, int fftSize, int hopSize, int bandwidth,
                   double threshold, int minTrackLength, double magWeight,
                   double freqWeight)
      : mWindowSize(windowSize),
        mWindow(windowFuncs[WindowType::kHann](windowSize)), mFFTSize(fftSize),
        mBins(fftSize / 2 + 1), mFFT(fftSize), mBandwidth(bandwidth),
        mThreshold(threshold), mMinTrackLength(minTrackLength),
        mMagWeight(magWeight), mFreqWeight(freqWeight), mCurrentFrame(0) {
    mWindowTransform = computeWindowTransform(mWindow);
    mOnes = VectorXd::Ones(mBandwidth);
    mWNorm = mWindowTransform.square().sum();
  }

  void processFrame(const ComplexVector &in, ComplexMatrix out) {
    using ArrayXcdMap = Map<const Array<std::complex<double>, Dynamic, RowMajor>>;
    using fluid::eigenmappings::ArrayXXcdToFluid;
    using Eigen::ArrayXXcd;

    const auto &epsilon = std::numeric_limits<double>::epsilon;
    ArrayXcdMap frame(in.data(), mBins);
    mBuf.push(frame);
    ArrayXd mag = frame.abs().real();
    ArrayXd correlation = getWindowCorrelation(mag);
    vector<int> peaks = findPeaks(correlation, mThreshold);
    peakContinuation(mTracks, peaks, mag);
    vector<SinePeak> sinePeaks = getActivePeaks(mTracks);
    ArrayXd frameSines = additiveSynthesis(sinePeaks);
    ArrayXXcd result(mBins, 2);
    if (mBuf.size() <= mMinTrackLength) {
      result.col(0) = ArrayXd::Zero(mBins);
      result.col(1) = ArrayXd::Zero(mBins);
    } else {
      ArrayXcd resultFrame = mBuf.front();
      ArrayXd resultMag = resultFrame.abs().real();
      ArrayXd frameResidual = resultMag - frameSines;
      frameResidual = (frameResidual < 0).select(0, frameResidual);
      ArrayXd all = frameSines + frameResidual;
      ArrayXd mult = (1.0 / all.max(epsilon()));
      result.col(0) = resultFrame * (frameSines * mult).min(1.0);
      result.col(1) = resultFrame * (frameResidual * mult).min(1.0);
      mBuf.pop();
    }
    auto iterator = std::remove_if(mTracks.begin(), mTracks.end(), [&](SineTrack track) {
      return (track.endFrame >= 0 && track.endFrame <= mCurrentFrame - mMinTrackLength);
    });
    mTracks.erase(iterator, mTracks.end());
    out = ArrayXXcdToFluid(result)();
    mCurrentFrame++;
  }

  void reset() { mCurrentFrame = 0; }

  void setThreshold(double threshold) {
    assert(0 <= threshold <= 1);
    mThreshold = threshold;
  }

  void setMinTrackLength(int minTrackLength) {
    mMinTrackLength = minTrackLength;
  }

  void setMagWeight(double magWeight) {
    assert(0 <= magWeight <= 1);
    mMagWeight = magWeight;
  }

  void setFreqWeight(double freqWeight) {
    assert(0 <= freqWeight <= 1);
    mFreqWeight = freqWeight;
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
  std::queue<ArrayXcd> mBuf;

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
    for (int i = idx, j = 0; i < std::min(idx + halfBW, mBins - 1); i++, j++) {
      sine[i] = amp * mWindowTransform(halfBW + j);
    }
    for (int i = idx, j = 0; i > std::max(idx - halfBW, 0); i--, j++) {
      sine[i] = amp * mWindowTransform(halfBW - j);
    }
    return sine;
  }
};
} // namespace rtsineextraction
} // namespace fluid
