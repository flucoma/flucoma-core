#pragma once

#include "algorithms/public/WindowFuncs.hpp"
#include "algorithms/util/ConvolutionTools.hpp"
#include "algorithms/util/FFT.hpp"
#include "algorithms/util/FluidEigenMappings.hpp"
#include "data/FluidIndex.hpp"
#include "data/TensorTypes.hpp"
#include <Eigen/Core>
#include <cmath>

namespace fluid {
namespace algorithm {

struct SpectralMass {
  index startBin;
  index centerBin;
  index endBin;
  double mass;
};

class OptimalTransport {

  using ArrayXd = Eigen::ArrayXd;
  using VectorXd = Eigen::VectorXd;
  using ArrayXXd = Eigen::ArrayXXd;
  using ArrayXcd = Eigen::ArrayXcd;
  template <typename T> using Ref = Eigen::Ref<T>;
  using TransportMatrix = std::vector<std::tuple<index, index, double>>;
  template <typename T> using vector = std::vector<T>;

public:
  void init(ArrayXd A, ArrayXd B) {
    mA = A;
    mB = B;
    mS1 = segmentSpectrum(A);
    mS2 = segmentSpectrum(B);
    mTransportMatrix = computeTransportMatrix(mS1, mS2);
    mInitialized = true;
  }

  bool initialized() { return mInitialized; }

  vector<SpectralMass> segmentSpectrum(const Ref<ArrayXd> magnitude) {
    const auto &epsilon = std::numeric_limits<double>::epsilon();
    vector<SpectralMass> masses;
    ArrayXd mag = magnitude;
    double totalMass = mag.sum() + epsilon;
    ArrayXd invMag = mag * (-1);
    vector<index> peaks = findPeaks(mag);
    vector<index> valleys = findPeaks(invMag);
    if (peaks.size() == 0 || valleys.size() == 0)
      return masses;
    index nextValley = valleys[0] > peaks[0] ? 0 : 1;
    SpectralMass firstMass{0, peaks[0], valleys[asUnsigned(nextValley)], 0};
    firstMass.mass =
        magnitude.segment(0, valleys[asUnsigned(nextValley)]).sum() / totalMass;
    masses.emplace_back(firstMass);
    for (index i = 1; asUnsigned(i) < peaks.size() - 1; i++) {
      index start = valleys[asUnsigned(nextValley)];
      if (start < 0)
        start = 0;
      index center = peaks[asUnsigned(i)];
      index end = valleys[asUnsigned(nextValley) + 1];
      if (end > magnitude.size() - 1)
        end = magnitude.size() - 1;
      if (end < 0 || start > end)continue;
      double mass = magnitude.segment(start, end - start).sum() / totalMass;
      masses.emplace_back(SpectralMass{start, center, end, mass});
      nextValley++;
    }
    index lastStart = valleys[asUnsigned(nextValley)];
    index lastSize = magnitude.size() - 1 - lastStart;
    if (lastSize < 0)
      lastSize = 0;
    if (lastSize > magnitude.size() - lastStart - 1)
      lastSize = magnitude.size() - lastStart - 1;
    double lastMass = magnitude.segment(lastStart, lastSize).sum();
    lastMass /= totalMass;
    masses.push_back(SpectralMass{valleys.at(asUnsigned(nextValley)),
                                  peaks.at(peaks.size() - 1),
                                  magnitude.size() - 1, lastMass});
    return masses;
  }

  TransportMatrix computeTransportMatrix(std::vector<SpectralMass> m1,
                                         std::vector<SpectralMass> m2) {
    TransportMatrix matrix;
    index index1 = 0, index2 = 0;
    double mass1 = m1[0].mass;
    double mass2 = m2[0].mass;
    while (true) {
      if (mass1 < mass2) {
        matrix.emplace_back(index1, index2, mass1);
        mDistance+=mass1*std::pow(index1 - index2, 2);
        mass2 -= mass1;

        index1++;
        if (index1 >= asSigned(m1.size()))
          break;
        mass1 = m1[asUnsigned(index1)].mass;
      } else {
        matrix.emplace_back(index1, index2, mass2);
        mDistance+=mass2*std::pow(index1 - index2, 2);
        mass1 -= mass2;
        index2++;
        if (index2 >= asSigned(m2.size()))
          break;
        mass2 = m2[asUnsigned(index2)].mass;
      }
    }
    return matrix;
  }

  void placeMass(const SpectralMass mass, index bin, double scale,
                 Ref<ArrayXd> input, Ref<ArrayXd> output) {
    for (index i = mass.startBin; i < mass.endBin; i++) {
      index pos = i + bin - mass.centerBin;
      if (pos < 0 || pos >= output.size())
        continue;
      double mag = scale * std::abs(input(i));
      output(pos) += mag;
    }
  }

  void interpolate(double interpolation, Eigen::Ref<ArrayXd> out) {
    for (auto t : mTransportMatrix) {
      SpectralMass m1 = mS1[asUnsigned(std::get<0>(t))];
      SpectralMass m2 = mS2[asUnsigned(std::get<1>(t))];
      index interpolatedBin = std::lrint((1 - interpolation) * m1.centerBin +
                                         interpolation * m2.centerBin);
      double interpolationFactor = interpolation;
      if (m1.centerBin != m2.centerBin) {
        interpolationFactor = ((double)interpolatedBin - (double)m1.centerBin) /
                              ((double)m2.centerBin - (double)m1.centerBin);
      }
      placeMass(m1, interpolatedBin,
                (1 - interpolation) * std::get<2>(t) / m1.mass, mA, out);
      placeMass(m2, interpolatedBin, interpolation * std::get<2>(t) / m2.mass,
                mB, out);
    }
  }

  std::vector<index> findPeaks(const Ref<ArrayXd> correlation) {
    std::vector<index> peaks;
    for (index i = 1; i < correlation.size() - 1; i++) {
      if (correlation(i) > correlation(i - 1) &&
          correlation(i) > correlation(i + 1)) {
        peaks.push_back(i);
      }
    }
    return peaks;
  }

  bool mInitialized{false};
  TransportMatrix mTransportMatrix;
  ArrayXd mA;
  ArrayXd mB;
  std::vector<SpectralMass> mS1;
  std::vector<SpectralMass> mS2;
  double mDistance{0};
};
} // namespace algorithm
} // namespace fluid
