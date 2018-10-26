#pragma once

#include "algorithms/ConvolutionTools.hpp"
#include "algorithms/FFT.hpp"
#include "algorithms/Windows.hpp"
#include "data/FluidTensor.hpp"
#include <Eigen/Core>

namespace fluid {
namespace algorithm {

using algorithm::correlateReal;
using algorithm::kEdgeWrapCentre;
using Eigen::Array;
using Eigen::ArrayXcd;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::Dynamic;
using Eigen::Map;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::RowMajor;
using Eigen::VectorXd;
using algorithm::FFT;

using std::vector;
using algorithm::windowFuncs;
using algorithm::WindowType;

struct SinePeak {
  double centerBin;
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

struct SinesPlusNoiseModel {
  using RealMatrix = FluidTensor<double, 2>;
  RealMatrix sines;
  RealMatrix noise;
};

class SineExtraction {
public:
  using RealMatrix = FluidTensor<double, 2>;

  SineExtraction(int windowSize, int fftSize, int hopSize, int bandwidth,
                 double threshold, int minTrackLength, double magWeight,
                 double freqWeight)
      : mWindowSize(windowSize),
        mWindow(windowFuncs[WindowType::kHann](windowSize)), mFFTSize(fftSize),
        mFFT(fftSize), mBandwidth(bandwidth), mThreshold(threshold),
        mMinTrackLength(minTrackLength), mMagWeight(magWeight),
        mFreqWeight(freqWeight) {
    mWindowTransform = computeWindowTransform(mWindow);
    mOnes = VectorXd::Ones(mBandwidth);
    mWNorm = mWindowTransform.square().sum();
  }

  const SinesPlusNoiseModel process(const RealMatrix &X) {
    using MatrixXdMap = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>;
    using ArrayXXdMap =
        Map<const Eigen::Array<double, Dynamic, Dynamic, RowMajor>>;
    using ArrayXdMap = Map<const Array<double, Dynamic, RowMajor>>;

    int nFrames = X.rows();
    int nBins = X.cols();
    ArrayXXdMap input(X.data(), nFrames, nBins);
    SinesPlusNoiseModel result;
    vector<SineTrack> tracks;
    result.sines = RealMatrix(nFrames, nBins);
    result.noise = RealMatrix(nFrames, nBins);
    MatrixXd sines(nFrames, nBins);
    MatrixXd noise(nFrames, nBins);

    for (int i = 0; i < nFrames; i++) {
      ArrayXdMap frame(X.row(i).data(), nBins);
      ArrayXd correlation = getWindowCorrelation(frame);
      vector<int> peaks = findPeaks(correlation, mThreshold);
      peakContinuation(tracks, peaks, frame, i);
    }
    vector<vector<SinePeak>> sinePeaks = filterTracks(tracks, nFrames);

    for (int i = 0; i < nFrames; i++) {
      ArrayXd frameSines = additiveSynthesis(sinePeaks[i]);
      ArrayXd frameResidual = ArrayXdMap(X.row(i).data(), nBins) - frameSines;
      frameResidual = (frameResidual < 0).select(0, frameResidual);
      sines.row(i) = frameSines;
      noise.row(i) = frameResidual;
    }
    MatrixXdMap(result.sines.data(), nFrames, nBins) = sines;
    MatrixXdMap(result.noise.data(), nFrames, nBins) = noise;

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
  int mMinTrackLength;
  double mMagWeight;
  double mFreqWeight;

  const void peakContinuation(vector<SineTrack> &tracks, vector<int> peaks,
                              ArrayXd frame, int frameNum) {
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
      sinePeaks.push_back(SinePeak{static_cast<double>(p), frame[p]});
    }

    for (auto &track : tracks) {
      if (track.active) {
        for (auto &&peak : sinePeaks) {
          double dist =
              mFreqWeight *
                  abs(log(track.peaks.back().centerBin / peak.centerBin)) +
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
    for (auto &&peak : sinePeaks) {
      if (!peak.assigned) {
        tracks.push_back(
            SineTrack{vector<SinePeak>{peak}, frameNum, -1, true, true});
      }
    }
    for (auto &&track : tracks) {
      if (track.active && !track.assigned) {
        track.active = false;
        track.endFrame = frameNum;
      }
    }
  }

  vector<vector<SinePeak>> filterTracks(vector<SineTrack> tracks, int nFrames) {
    vector<vector<SinePeak>> sinePeaks(nFrames, vector<SinePeak>());
    for (auto &&track : tracks) {
      if (track.endFrame - track.startFrame >= mMinTrackLength) {
        for (int i = track.startFrame; i < track.endFrame; i++) {
          sinePeaks[i].push_back(track.peaks[i - track.startFrame]);
        }
      }
    }
    return sinePeaks;
  }

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
    ArrayXd result = ArrayXd::Zero(mBandwidth);
    ArrayXcd transform = mFFT.process(Map<ArrayXd>(window.data(), mWindowSize));
    for (int i = 0; i < halfBW; i++) {
      result(halfBW + i) = result(halfBW - i) = abs(transform(i));
    }
    return result;
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

  /*ArrayXd additiveSynthesis(vector<int> peaks, ArrayXd spectrum) {
    ArrayXd result = ArrayXd::Zero(mFFTSize / 2 + 1);
    for (auto &p : peaks) {
      result += synthesizePeak(p, spectrum(p));
    }
    return result;
  }*/

  ArrayXd additiveSynthesis(vector<SinePeak> peaks) {
    ArrayXd result = ArrayXd::Zero(mFFTSize / 2 + 1);
    for (auto &p : peaks) {
      result += synthesizePeak(p.centerBin, p.mag);
    }
    return result;
  }

  ArrayXd synthesizePeak(int idx, double amp) {
    int halfBW = mBandwidth / 2;
    ArrayXd sine = ArrayXd::Zero(mFFTSize / 2 + 1);
    int frameSize = mFFTSize / 2 + 1;
    for (int i = idx, j = 0; i < std::min(idx + halfBW, frameSize - 1); i++, j++) {
      sine[i] = amp * mWindowTransform(halfBW + j);
    }
    for (int i = idx, j = 0; i > std::max(idx - halfBW, 0); i--, j++) {
      sine[i] = amp * mWindowTransform(halfBW - j);
    }
    return sine;
  }
};
} // namespace algorithm
} // namespace fluid
