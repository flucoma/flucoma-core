//  Based on https://github.com/sportdeath/audio_transport/
// T.Henderson and J.Solomon, Audio Transport:
// a generalized portamento via optimal transport.
// Proceedings of DAFX 2019.

#pragma once

#include "algorithms/public/WindowFuncs.hpp"
#include "algorithms/util/ConvolutionTools.hpp"
#include "algorithms/util/FFT.hpp"
#include "algorithms/public/STFT.hpp"
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
  using ArrayXi = Eigen::ArrayXi;
  using ArrayXcd = Eigen::ArrayXcd;
  template <typename T> using Ref = Eigen::Ref<T>;
  using TransportMatrix = std::vector<std::tuple<index, index, double>>;
  template <typename T> using vector = std::vector<T>;

public:

  AudioTransport(index maxFFTSize)
      : mWindowSize(maxFFTSize), mFFTSize(maxFFTSize),
        mBins(maxFFTSize / 2 + 1), mFFT(maxFFTSize),
        mSTFT(maxFFTSize, maxFFTSize, maxFFTSize / 2),
        mISTFT(maxFFTSize, maxFFTSize, maxFFTSize / 2),
        mReassignSTFT(maxFFTSize, maxFFTSize, maxFFTSize /2) {}

  void init(index windowSize, index fftSize, index hopSize) {
    mWindowSize = windowSize;
    mWindow = ArrayXd::Zero(mWindowSize);
    WindowFuncs::map()[WindowFuncs::WindowTypes::kHann](mWindowSize,mWindow);
    mWindowSquared = mWindow * mWindow;
    mFFTSize = fftSize;
    mHopSize = hopSize;
    mBins = fftSize / 2 + 1;
    mPhase = ArrayXd::Zero(mBins);
    mChanged = ArrayXi::Zero(mBins);
    mBinFreqs = ArrayXd::LinSpaced(mBins, 0, mBins - 1) * (2 * pi) / mFFTSize;
    mPhaseDiff = mBinFreqs * mHopSize;
    mSTFT = STFT(windowSize, fftSize, hopSize);
    mISTFT = ISTFT(windowSize, fftSize, hopSize);
    mReassignSTFT = STFT(windowSize, fftSize, hopSize,
        static_cast<index>(WindowFuncs::WindowTypes::kHannD));
    mInitialized = true;
  }

  bool initialized() const{return mInitialized;}

  void processFrame(RealVectorView in1, RealVectorView in2,
                    double weight, RealMatrixView out) {
    using namespace _impl;
    using namespace Eigen;
    assert(mInitialized);
    ArrayXd frame1 = asEigen<Array>(in1);
    ArrayXd frame2 = asEigen<Array>(in2);
    ArrayXcd spectrum1(mBins);
    ArrayXcd spectrum1Dh(mBins);
    ArrayXcd spectrum2(mBins);
    ArrayXcd spectrum2Dh(mBins);
    ArrayXd output(frame1.size());
    mSTFT.processFrame(frame1, spectrum1);
    mReassignSTFT.processFrame(frame1, spectrum1Dh);
    mSTFT.processFrame(frame2, spectrum2);
    mReassignSTFT.processFrame(frame2, spectrum2Dh);
    ArrayXcd result = interpolate(
        spectrum1, spectrum1Dh,
        spectrum2, spectrum2Dh, weight
    );
    mISTFT.processFrame(result, output);
    out.row(0) = asFluid(output);
    out.row(1) = asFluid(mWindowSquared);
  }

  vector<SpetralMass> segmentSpectrum(
    const Ref<ArrayXd> mag,
    const Ref<ArrayXd> reasignedFreq) {

    vector<SpetralMass> masses;
    double totalMass = mag.sum() + epsilon;
    ArrayXi sign = (reasignedFreq > mBinFreqs).cast<int>();
    mChanged.setZero();
    mChanged.segment(1, mBins - 1) =
        sign.segment(1, mBins - 1) - sign.segment(0, mBins - 1);
    SpetralMass currentMass{0, 0, 0, 0};
    for(index i = 1; i < mChanged.size(); i++){
      if(mChanged(i) == -1){
        double d1 = reasignedFreq(i - 1) - mBinFreqs(i - 1);
        double d2 = mBinFreqs(i) - reasignedFreq(i);
        currentMass.centerBin  =  d1<d2? i - 1:i;
      }
      if(mChanged(i) == 1){
        currentMass.endBin = i;
        currentMass.mass = mag.segment(
          currentMass.startBin, i - currentMass.startBin
        ).sum() / totalMass;
        masses.emplace_back(currentMass);
        currentMass = SpetralMass{i, i, i, 0};
      }
    }
    currentMass.endBin = mBins;
    currentMass.mass = mag.segment(currentMass.startBin,
                       mBins - currentMass.startBin).sum() / totalMass;
    masses.emplace_back(currentMass);
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

  ArrayXcd interpolate(Ref<ArrayXcd> in1, Ref<ArrayXcd> in1Dh,
                      Ref<ArrayXcd> in2, Ref<ArrayXcd> in2Dh,
                       double interpolation) {
    ArrayXd mag1 = in1.abs().real();
    ArrayXd mag2 = in2.abs().real();
    ArrayXcd result = ArrayXcd::Zero(mBins);
    double mag1Sum = mag1.sum();
    double mag2Sum = mag2.sum();
    if(mag1Sum <= 0 && mag2Sum <= 0){return result;}
    else if(mag1Sum > 0 && mag2Sum <= 0){return in1;}
    else if(mag1Sum <= 0 && mag2Sum > 0){return in2;}
    ArrayXd phase1 = in1.arg().real();
    ArrayXd phase2 = in2.arg().real();
    ArrayXd reasignedW1 = mBinFreqs - (in1Dh / in1).imag();
    ArrayXd reasignedW2 = mBinFreqs - (in2Dh / in2).imag();
    ArrayXd newAmplitudes = ArrayXd::Zero(mBins);
    ArrayXd newPhases = ArrayXd::Zero(mBins);
    std::vector<SpetralMass> s1 = segmentSpectrum(mag1, reasignedW1);
    std::vector<SpetralMass> s2 = segmentSpectrum(mag2, reasignedW2);
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
              (1 - interpolationFactor) * reasignedW1(m1.centerBin) +
              interpolationFactor * reasignedW2(m2.centerBin);
      double nextPhase = mPhase(interpolatedBin) + interpolatedFreq * mHopSize;
      double centerPhase = nextPhase - mPhaseDiff(interpolatedBin);
      placeMass(m1, interpolatedBin,
                (1 - interpolation) * std::get<2>(t) / m1.mass, centerPhase,
                in1, result, nextPhase, newAmplitudes, newPhases);
      placeMass(m2, interpolatedBin, interpolation * std::get<2>(t) / m2.mass,
                centerPhase, in2, result, nextPhase, newAmplitudes, newPhases);
    }
    mPhase = newPhases;
    return result;
  }

  index mWindowSize{1024};
  index mHopSize{512};
  ArrayXd mBinFreqs;
  ArrayXd mWindow;
  ArrayXd mWindowSquared;
  index mFFTSize{1024};
  index mBins{513};
  FFT mFFT;
  bool mInitialized{false};
  ArrayXd mPhase;
  ArrayXd mPhaseDiff;
  ArrayXi mChanged;
  STFT mSTFT;
  ISTFT mISTFT;
  STFT mReassignSTFT;
};
} // namespace algorithm
} // namespace fluid
