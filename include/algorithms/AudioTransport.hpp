//  Based on https://github.com/sportdeath/audio_transport/
// T.Henderson and J.Solomon, Audio Transport:
// a generalized portamento via optimal transport.
// Proceedings of DAFX 2019.

#pragma once

#include "algorithms/public/WindowFuncs.hpp"
#include "algorithms/util/ConvolutionTools.hpp"
#include "algorithms/util/FFT.hpp"
#include "algorithms/util/FluidEigenMappings.hpp"
#include "data/TensorTypes.hpp"
#include "data/FluidIndex.hpp"
#include <Eigen/Core>
#include <cmath>

namespace fluid {
namespace algorithm {

struct SpetralMass {
  index startBin;
  index centerBin;
  index endBin;
  double mass;
};

class AudioTransport {

  using ArrayXd = Eigen::ArrayXd;
  using VectorXd = Eigen::VectorXd;
  using ArrayXXd = Eigen::ArrayXXd;
  using ArrayXcd = Eigen::ArrayXcd;
  template <typename T> using Ref = Eigen::Ref<T>;
  using TransportMatrix = std::vector<std::tuple<index, index, double>>;
  template <typename T> using vector = std::vector<T>;

public:

  AudioTransport(index maxFFTSize)
      : mWindowSize(maxFFTSize), mFFTSize(maxFFTSize),
        mBins(maxFFTSize / 2 + 1), mFFT(maxFFTSize) {}

  void init(index windowSize, index fftSize, index hopSize, index bandwidth) {
    mWindowSize = windowSize;
    mBandwidth = bandwidth;
    mWindow = ArrayXd::Zero(mWindowSize);
    mOnes = VectorXd::Ones(mBandwidth);
    WindowFuncs::map()[WindowFuncs::WindowTypes::kHann](mWindowSize,mWindow);
    computeWindowTransform(mWindow);
    mWNorm = mWindowTransform.square().sum();
    mFFTSize = fftSize;
    mHopSize = hopSize;
    mBins = fftSize / 2 + 1;
    mFFT.resize(fftSize);
    mPhase = ArrayXd::Zero(mBins);
    mPrevPhase1 = ArrayXd::Zero(mBins);
    mPrevPhase2 = ArrayXd::Zero(mBins);
    mBinIndices = ArrayXd::LinSpaced(mBins, 0, mBins - 1);
    mInitialized = true;
  }

  bool initialized(){return mInitialized;}
  void processFrame(const ComplexVectorView in1, const ComplexVectorView in2,
                    double weight, ComplexVectorView out) {
    using namespace _impl;
    using namespace Eigen;
    assert(mInitialized);
    ArrayXcd frame1 = asEigen<Array>(in1);
    ArrayXcd frame2 = asEigen<Array>(in2);
    ArrayXcd result(mBins);
    result = interpolate(frame1, frame2, weight);
    out = asFluid(result);
  }

  vector<SpetralMass> segmentSpectrum(const Ref<ArrayXd> magnitude) {
    const auto &epsilon = std::numeric_limits<double>::epsilon();
    vector<SpetralMass> masses;
    ArrayXd mag = magnitude;
    double totalMass = mag.sum() + epsilon;
    ArrayXd correlation = getWindowCorrelation(mag);
    correlation = correlation / correlation.maxCoeff();
    ArrayXd invCorrelation = correlation * (-1);
    vector<index> peaks = findPeaks(correlation, 0);
    vector<index> valleys = findPeaks(invCorrelation, -1);
    if(peaks.size()==0 || valleys.size()==0) return masses;
    index nextValley = valleys[0] > peaks[0] ? 0 : 1;
    SpetralMass firstMass{0, peaks[0], valleys[asUnsigned(nextValley)], 0};
    firstMass.mass =
        magnitude.segment(0, valleys[asUnsigned(nextValley)]).sum() / totalMass;
    masses.emplace_back(firstMass);
    for (index i = 1; asUnsigned(i) < peaks.size() - 1; i++) {
      index start = valleys[asUnsigned(nextValley)];
      index center = peaks[asUnsigned(i)];
      index end = valleys[asUnsigned(nextValley) + 1];
      double mass = magnitude.segment(start, end - start).sum() / totalMass;
      masses.emplace_back(SpetralMass{start, center, end, mass});
      nextValley++;
    }

    double lastMass = magnitude
                          .segment(valleys[asUnsigned(nextValley)],
                                   magnitude.size() - 1 - valleys[asUnsigned(nextValley)])
                          .sum();

    lastMass /= totalMass;
    masses.push_back(SpetralMass{valleys.at(asUnsigned(nextValley)),
                                 peaks.at(peaks.size() - 1), mBins - 1,
                                 lastMass});
    return masses;
  }

  TransportMatrix computeTransportMatrix(std::vector<SpetralMass> m1,
                                         std::vector<SpetralMass> m2) {
    TransportMatrix matrix;
    index index1 = 0, index2 = 0;
    double mass1 = m1[0].mass;
    double mass2 = m2[0].mass;
    while (true) {
      if (mass1 < mass2) {
        matrix.emplace_back(index1, index2, mass1);
        mass2 -= mass1;
        index1++;
        if (index1 >= asSigned(m1.size()))
          break;
        mass1 = m1[asUnsigned(index1)].mass;
      } else {
        matrix.emplace_back(index1, index2, mass2);
        mass1 -= mass2;
        index2++;
        if (index2 >= asSigned(m2.size()))
          break;
        mass2 = m2[asUnsigned(index2)].mass;
      }
    }
    return matrix;
  }

  void placeMass(const SpetralMass mass, index bin, double scale,
                 double centerPhase, Ref<ArrayXcd> input, Ref<ArrayXcd> output,
                 double nextPhase, Ref<ArrayXd> amplitudes,
                 Ref<ArrayXd> phases) {
    double phaseShift = centerPhase - std::arg(input(mass.centerBin));
    for (index i = mass.startBin; i < mass.endBin; i++) {
      index pos = i + bin - mass.centerBin;
      if (pos < 0 || pos >= output.size())
        continue;
      double phase = phaseShift + std::arg(input(i));
      double mag = scale * std::abs(input(i));
      output(pos) += std::polar(mag, phase);
      if (mag > amplitudes(pos)) {
        amplitudes(pos) = mag;
        phases(pos) = nextPhase;
      }
    }
  }

  ArrayXcd interpolate(Eigen::Ref<ArrayXcd> in1, Eigen::Ref<ArrayXcd> in2,
                       double interpolation) {
    ArrayXd mag1 = in1.abs().real();
    ArrayXd mag2 = in2.abs().real();
    ArrayXcd result = ArrayXcd::Zero(in1.size());
    double mag1Sum = mag1.sum();
    double mag2Sum = mag2.sum();

    if(mag1Sum <= 0 && mag2Sum <= 0){return result;}
    else if(mag1Sum > 0 && mag2Sum <= 0){return in1;}
    else if(mag1Sum <= 0 && mag2Sum > 0){return in2;}

    ArrayXd phase1 = in1.arg().real();
    ArrayXd phase2 = in2.arg().real();
    ArrayXd instFreq1 = instFreq(phase1, mPrevPhase1);
    ArrayXd instFreq2 = instFreq(phase2, mPrevPhase2);
    ArrayXd newAmplitudes = ArrayXd::Zero(mBins);
    ArrayXd newPhases = ArrayXd::Zero(mBins);

    std::vector<SpetralMass> s1 = segmentSpectrum(mag1);
    std::vector<SpetralMass> s2 = segmentSpectrum(mag2);
    if(s1.size() == 0 || s2.size()==0){
      return result;
    }

    TransportMatrix matrix = computeTransportMatrix(s1, s2);
    for (auto t : matrix) {
      SpetralMass m1 = s1[asUnsigned(std::get<0>(t))];
      SpetralMass m2 = s2[asUnsigned(std::get<1>(t))];
      index interpolatedBin = std::lrint((1 - interpolation) * m1.centerBin +
                                       interpolation * m2.centerBin);
      double interpolationFactor = interpolation;
      if (m1.centerBin != m2.centerBin) {
        interpolationFactor = ((double)interpolatedBin - (double)m1.centerBin) /
                              ((double)m2.centerBin - (double)m1.centerBin);
      }
      double interpolatedFreq =
          (1 - interpolationFactor) * instFreq1(m1.centerBin) +
          interpolationFactor * instFreq2(m2.centerBin);
      double centerPhase = mPhase(interpolatedBin) +
                           (interpolatedFreq * mWindowSize / 2.) / 2. -
                           (M_PI * interpolatedBin);
      double nextPhase = centerPhase +
                         (interpolatedFreq * mWindowSize / 2.) / 2. +
                         (M_PI * interpolatedBin);
      placeMass(m1, interpolatedBin,
                (1 - interpolation) * std::get<2>(t) / m1.mass, centerPhase,
                in1, result, nextPhase, newAmplitudes, newPhases);
      placeMass(m2, interpolatedBin, interpolation * std::get<2>(t) / m2.mass,
                centerPhase, in2, result, nextPhase, newAmplitudes, newPhases);
    }
    mPhase = newPhases;
    mPrevPhase1 = phase1;
    mPrevPhase2 = phase2;
    return result;
  }

  // TODO: refactor along with SineExtraction
  ArrayXd getWindowCorrelation(const Ref<ArrayXd> frame) {
    ArrayXd squareMag = frame.square();
    ArrayXd corr(frame.size());
    ArrayXd spectrumNorm(frame.size());
    correlateReal(corr.data(), frame.data(), asUnsigned(frame.size()),
                  mWindowTransform.data(), asUnsigned(mBandwidth), kEdgeWrapCentre);
    convolveReal(spectrumNorm.data(), squareMag.data(), asUnsigned(frame.size()),
                 mOnes.data(), asUnsigned(mBandwidth), kEdgeWrapCentre);
    corr = corr.square() / (spectrumNorm * mWNorm);
    return corr;
  }

  void computeWindowTransform(Ref<ArrayXd> window) {
    index halfBW = mBandwidth / 2;
    mWindowTransform = ArrayXd::Zero(mBandwidth);
    ArrayXcd transform =
        mFFT.process(Eigen::Map<ArrayXd>(window.data(), mWindowSize));
    for (index i = 0; i < halfBW; i++) {
      mWindowTransform(halfBW + i) = mWindowTransform(halfBW - i) =
          abs(transform(i));
    }
  }

  std::vector<index> findPeaks(const Ref<ArrayXd> correlation, double threshold) {
    std::vector<index> peaks;
    for (index i = 1; i < correlation.size() - 1; i++) {
      if (correlation(i) > correlation(i - 1) &&
          correlation(i) > correlation(i + 1) && correlation(i) > threshold) {
        peaks.push_back(i);
      }
    }
    return peaks;
  }

  ArrayXd princArg(Ref<ArrayXd> phase) {
    ArrayXd result = M_PI + (phase + M_PI);
    result = result.unaryExpr([](const double x) { return fmod(x, -(2 * M_PI)); });
    return result;
  }

  ArrayXd instFreq(Ref<ArrayXd> phase, Ref<ArrayXd> prevPhase) {
    ArrayXd w = mBinIndices * (2 * M_PI) / mFFTSize;
    ArrayXd dPhase = w * mHopSize;
    ArrayXd phaseInc = phase - prevPhase - dPhase;
    phaseInc = phaseInc - (2 * M_PI) * (phaseInc / (2 * M_PI)).round();
    ArrayXd instW = w + phaseInc / mHopSize;
    return instW;
  }

  index mBandwidth{76};
  index mWindowSize{1024};
  index mHopSize{512};
  ArrayXd mBinIndices;
  ArrayXd mWindow;
  ArrayXd mWindowTransform;
  index mFFTSize{1024};
  index mBins{513};
  FFT mFFT;
  bool mInitialized{false};
  ArrayXd mPhase;
  ArrayXd mPrevPhase1;
  ArrayXd mPrevPhase2;
  VectorXd mOnes;
  double mWNorm{1.0};
};
} // namespace algorithm
} // namespace fluid
