#pragma once

#include "../util/AlgorithmUtils.hpp"
#include "../../data/TensorTypes.hpp"
#include "../util/ConvolutionTools.hpp"
#include "../util/FFT.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "WindowFuncs.hpp"

#include <Eigen/Core>
#include <queue>

namespace fluid {
namespace algorithm {


struct SinePeak {
  // int centerBin;
  double freq;
  double logMag;
  bool assigned;
};

struct SineTrack {
  std::vector<SinePeak> peaks;
  int startFrame;
  int endFrame;
  bool active;
  bool assigned;
};

class SineExtraction {
  using ArrayXd = Eigen::ArrayXd;
  using VectorXd = Eigen::VectorXd;
  using ArrayXcd = Eigen::ArrayXcd;
  template<typename T>using vector = std::vector<T>;

public:

  SineExtraction(int maxFFTSize)
      : mWindowSize(maxFFTSize), mFFTSize(maxFFTSize),
        mBins(maxFFTSize / 2 + 1), mFFT(maxFFTSize), mBandwidth(maxFFTSize / 16) {
  }

  void init(int windowSize, int fftSize, int hopSize, int bandwidth,
                   double threshold, int minTrackLength, double magWeight,
                   double freqWeight){
    mWindowSize = windowSize;
    mWindow = ArrayXd::Zero(mWindowSize);
    WindowFuncs::map()[WindowFuncs::WindowTypes::kHann](mWindowSize, mWindow);
    mFFTSize = fftSize;
    mBins = fftSize / 2 + 1;
    mFFT.resize(fftSize);
    mBandwidth = bandwidth;
    mThreshold = threshold;
    mMinTrackLength = minTrackLength;
    mMagWeight = magWeight;
    mFreqWeight = freqWeight;
    mCurrentFrame = 0;
    mWindowTransform = computeWindowTransform(mWindow);
    mOnes = VectorXd::Ones(mBandwidth);
    mWNorm = mWindowTransform.square().sum();
    mTracks = vector<SineTrack>();
    mBuf = std::queue<ArrayXcd>();
    mInitialized = true;
  }

  void processFrame(const ComplexVectorView in, ComplexMatrixView out) {
    assert(mInitialized);
    using Eigen::Array;
    using Eigen::ArrayXXcd;
    const auto &epsilon = std::numeric_limits<double>::epsilon();
    ArrayXcd frame = _impl::asEigen<Array>(in);
    mBuf.push(frame);
    ArrayXd mag = frame.abs().real();
    ArrayXd logMag = 20 * mag.max(epsilon).log10();
    ArrayXd correlation = getWindowCorrelation(mag);
    vector<SinePeak> peaks = findPeaks(correlation, mThreshold, logMag);
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
      ArrayXd mult = (1.0 / all.max(epsilon));
      result.col(0) = resultFrame * (frameSines * mult).min(1.0);
      result.col(1) = resultFrame * (frameResidual * mult).min(1.0);
      mBuf.pop();
    }
    auto iterator =
        std::remove_if(mTracks.begin(), mTracks.end(), [&](SineTrack track) {
          return (track.endFrame >= 0 &&
                  track.endFrame <= mCurrentFrame - mMinTrackLength);
        });
    mTracks.erase(iterator, mTracks.end());
    out = _impl::asFluid(result);
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

  bool initialized(){return mInitialized;}

private:
  int mWindowSize;
  //vector<double> mWindow;
  ArrayXd mWindow;
  int mFFTSize;
  int mBins;
  FFT mFFT;
  VectorXd mOnes;
  ArrayXd mW;
  double mWNorm;
  int mBandwidth;
  double mThreshold{0.7};
  ArrayXd mWindowTransform;
  int mMinTrackLength{15};
  double mMagWeight{0.01};
  double mFreqWeight{0.5};
  size_t mCurrentFrame{0};
  vector<SineTrack> mTracks;
  std::queue<ArrayXcd> mBuf;
  bool mInitialized{false};

  const void peakContinuation(vector<SineTrack> &tracks,
                              vector<SinePeak> sinePeaks, const ArrayXd frame) {
    using namespace std;


    vector<tuple<double, SineTrack *, SinePeak *>> distances;
    for (auto &&track : tracks) {
      track.assigned = false;
    }

    for (auto &track : tracks) {
      if (track.active) {
        for (auto &&peak : sinePeaks) {
          double dist =
              mFreqWeight * abs(log(track.peaks.back().freq / peak.freq)) +
              mMagWeight * abs(track.peaks.back().logMag - peak.logMag);
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

  ArrayXd computeWindowTransform(ArrayXd window) {
    int halfBW = mBandwidth / 2;
    ArrayXd result = ArrayXd::Zero(mBandwidth);
    ArrayXcd transform = mFFT.process(Eigen::Map<ArrayXd>(window.data(), mWindowSize));
    for (int i = 0; i < halfBW; i++) {
      result(halfBW + i) = result(halfBW - i) = abs(transform(i));
    }
    return result;
  }

  vector<SinePeak> findPeaks(const ArrayXd correlation, double threshold,
                             const ArrayXd logMag) {
    vector<SinePeak> peaks;
    for (int i = 1; i < correlation.size() - 1; i++) {
      if (correlation(i) > correlation(i - 1) &&
          correlation(i) > correlation(i + 1) && correlation(i) > threshold) {
        double p =
            0.5 * (correlation(i - 1) - correlation(i + 1)) /
            (correlation(i - 1) - 2 * correlation(i) + correlation(i + 1));
        double interpFreq = i + p;
        double interpLogMag =
            logMag(i) - 0.25 * (correlation(i - 1) - correlation(i + 1)) * p;

        peaks.push_back(SinePeak{interpFreq, interpLogMag});
      }
    }
    return peaks;
  }

  ArrayXd additiveSynthesis(const vector<SinePeak> peaks) {
    ArrayXd result = ArrayXd::Zero(mBins);
    for (auto &p : peaks) {
      result += synthesizePeak(p);
    }
    return result;
  }

  ArrayXd synthesizePeak(SinePeak p) {
    int halfBW = mBandwidth / 2;
    ArrayXd sine = ArrayXd::Zero(mBins);
    int freqBin = std::round(p.freq);
    if(freqBin >= mBins - 1)freqBin = mBins - 1;
    double amp = std::pow(10, p.logMag / 20);

    for (int i = freqBin, j = 0; i < std::min(freqBin + halfBW, mBins - 1); i++, j++) {
      sine[i] = amp * mWindowTransform(halfBW + j);
    }
    for (int i = freqBin, j = 0; i > std::max(freqBin - halfBW, 0); i--, j++) {
      sine[i] = amp * mWindowTransform(halfBW - j);
    }
    return sine;
  }
};
} // namespace algorithm
} // namespace fluid
