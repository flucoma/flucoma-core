
#pragma once

#include "../data/FluidTensor.hpp"

#include <cassert>
#include <cmath>
#include <algorithm>

namespace fluid {
namespace descriptors {
  
class Descriptors
{
  using RealTensor = fluid::FluidTensor<double, 1>;
  using Real = fluid::FluidTensorView<double, 1>;

  // Helper structs
  
  struct NOP  { double operator()(double a) { return a; } };
  struct Pow2 { double operator()(double a) { return a * a; } };
  struct Pow3 { double operator()(double a) { return a * a * a; } };
  struct Pow4 { double operator()(double a) { return Pow2()(Pow2()(a)); } };
  struct Log  { double operator()(double a) { return log(a); } };
  struct Abs  { double operator()(double a) { return fabs(a); } };

  struct SqDifference { double operator()(double a, double b) { return (a-b) * (a-b); } };
  struct AbsDifference { double operator()(double a, double b) { return fabs(a-b); } };
  
  // For Log Spectral Difference
  
  struct LogDBDifference
  {
    double operator()(double a, double b)
    {
      const double difference = 20.0 * log10(a / b);
      return difference * difference;
    }
  };
  
  // Itakura-Saito difference
  
  struct ISDifference
  {
    double operator()(double a, double b)
    {
      double ratio = a / b;
      ratio *= ratio;
      return ratio - log(ratio) - 1;
    }
  };
  
  // Kullback-Liebler Difference (+ Symmetric and Modifed Versions)
  
  struct KLDifference { double operator()(double a, double b) { return a * log(a / b); } };
  struct SKLDifference { double operator()(double a, double b) { return (a - b) * log(a / b); } };
  struct MKLDifference { double operator()(double a, double b) { return log(a / b); } };
  
  // Index difference from a fixed reference
  
  template <typename Op>
  struct IndexDiff {
    IndexDiff(double value) : mValue(value) {}
    double operator()(double a) { return Op()(a - mValue); }
    double mValue;
  };
  
public:
  
  // Shape
  
  static double centroid(const Real& input)
  {
    return statIndexWeightedSum(input, NOP()) / statSum(input);
  }
  
  static double spread(const Real& input)
  {
    double c = centroid(input);
    return statIndexWeightedSum(input, IndexDiff<Pow2>(c)) / statSum(input);
  }
  
  static double skewness(const Real& input)
  {
    double c = centroid(input);
    double sN = sqrt(spread(input));
    return statIndexWeightedSum(input, IndexDiff<Pow3>(c)) / (sN * sN * sN * statSum(input));
  }
  
  static double kurtosis(const Real& input)
  {
    double c = centroid(input);
    double sN = spread(input);
    return statIndexWeightedSum(input, IndexDiff<Pow4>(c)) / (sN * sN * statSum(input));
  }
  
  static double flatness(const Real& input)
  {
    return statGeometricMean(input) / statMean(input);
  }
  
  static double rolloff(const Real& input, double centile = 95.0)
  {
    return statPDFPercentile(input, centile);
  }
  
  // Level
  
  static double RMS(const Real& input)
  {
    return sqrt(statMeanSquares(input));
  }
  
  static double crest(const Real& input)
  {
    return statMax(input, Abs()) / RMS(input);
  }
  
  static double peak(const Real& input)
  {
    return statMax(input, Abs());
  }
  
  // HFC
  
  // If handed a spectra returns the HFC with positions scaled in bins
  
  static double HFC(const Real& input)
  {
    return statIndexWeightedSum(input);
  }
  
  // Vector Differences
  
  // Normalise a vector (or spectral frame)
  
  static void normalise(Real& input, bool power)
  {
    const double sum = (power ? sqrt(statSumSquares(input)) : statSum(input));
    
    if (sum)
    {
      const double factor = 1.0 / sum;
      input.apply([&](double& x){x *= factor;});
    }
  }
  
  // Remove values from two vectors in positions that are not greater in the first vector than the second
  
  static void forwardFilter(Real& vec1, Real& vec2)
  {
    assert(vec1.size() == vec2.size() && "Vectors should match in size");

    for (auto it = vec1.begin(), jt = vec2.begin(); it != vec1.end(); it++, jt++)
    {
      if (*it < *jt)
        *it = *jt = 0.0;
    }
  }

  // L1 Norm difference between two vectors
  
  static double differenceL1Norm(const Real& vec1, const Real& vec2)
  {
    return statSum(vec1, vec2, AbsDifference());
  }
  
  // L2 Norm difference between two vectors
  
  static double differenceL2Norm(const Real& vec1, const Real& vec2)
  {
    return sqrt(statSum(vec1, vec2, SqDifference()));
  }
  
  // Log difference between two vectors (if spectra expected as amplitude spectra, not power spectra)
  
  static double differenceLog(const Real& vec1, const Real& vec2)
  {
    return sqrt(statSum(vec1, vec2, LogDBDifference()));
  }
  
  // Foote difference between two vectors
  
  static double differenceFT(const Real& vec1, const Real& vec2)
  {
    assert(vec1.size() == vec2.size() && "Vectors should match in size");
    
    return statSumProduct(vec1, vec2) / sqrt(statSumSquares(vec1) * statSumSquares(vec2));
  }
  
  // Itakura-Saito difference between two vectors (if spectra expected as amplitude spectra, not power spectra)

  static double differenceIS(const Real& vec1, const Real& vec2)
  {
    return statSum(vec1, vec2, ISDifference());
  }
  
  // Kullback-Liebler difference between two vectors (if spectra expected as amplitude spectra, not power spectra)
  
  static double differenceKL(const Real& vec1, const Real& vec2)
  {
    return statSum(vec1, vec2, KLDifference());
  }
  
  // Symmetric Kullback-Liebler difference between two vectors (if spectra expected as amplitude spectra, not power spectra)
  
  static double differenceSKL(const Real& vec1, const Real& vec2)
  {
    return 0.5 * statSum(vec1, vec2, SKLDifference());
  }
  
  // Modified Kullback-Liebler difference between two vectors (if spectra expected as amplitude spectra, not power spectra)
  
  static double differenceMKL(const Real& vec1, const Real& vec2)
  {
    return statSum(vec1, vec2, MKLDifference());
  }
  
private:

  // Sums
  
  template <typename Op>
  static double statSum(const Real& input, Op modifier)
  {
    double sum = 0.0;
    
    for (auto it = input.begin(); it != input.end(); it++)
      sum += modifier(*it);
    
    return sum;
  }
  
  template <typename Op>
  static double statIndexWeightedSum(const Real& data, Op indexModifier)
  {
    double sum = 0.0;
    int i = 0;
    
    for (auto it = data.begin(); it != data.end(); it++, i++)
      sum += indexModifier(i) * *it;
    
    return sum;
  }
  
  template <typename Op>
  static double statSum(const Real& v1, const Real& v2, Op modifier)
  {
    assert(v1.size() == v2.size() && "Vectors should match in size");
    
    double sum = 0.0;
  
    for (auto it = v1.begin(), jt = v2.begin(); it != v1.end(); it++, jt++)
      sum += modifier(*it, *jt);
    
    return sum;
  }
  
  static double statSum(const Real& input)
  {
    return statSum(input, NOP());
  }
  
  static double statSumSquares(const Real& input)
  {
    return statSum(input, Pow2());
  }
  
  static double statSumAbs(const Real& input)
  {
    return statSum(input, Abs());
  }
  
  static double statSumLogs(const Real& input)
  {
    return statSum(input, Log());
  }
  
  static double statIndexWeightedSum(const Real& input)
  {
    return statIndexWeightedSum(input, NOP());
  }
  
  static double statSumProduct(const Real& v1, const Real& v2)
  {
    return statSum(v1, v2, std::multiplies<double>());
  }
  
  // Means
  
  static double statMean(const Real& input)
  {
    return statSum(input) / input.size();
  }
  
  static double statMeanSquares(const Real& input)
  {
    return statSumSquares(input) / input.size();
  }
  
  static double statGeometricMean(const Real& input)
  {
    return exp(statSumLogs(input) / input.size());
  }

  // Min / Max Values
  
  template <typename Op>
  static double statMin(const Real& input, Op modifier)
  {
    double min = std::numeric_limits<double>::infinity();
    
    for (auto it = input.begin(); it != input.end(); it++)
      min = std::min(min, modifier(*it));
    
    return min;
  }
  
  template <typename Op>
  static double statMax(const Real& input, Op modifier)
  {
    double max = -std::numeric_limits<double>::infinity();
    
    for (auto it = input.begin(); it != input.end(); it++)
      max = std::max(max, modifier(*it));

    return max;
  }
  
  // PDF Percentile
  
  static double statPDFPercentile(const Real& input, double centile)
  {
    double target = statSum(input) * std::min(100.0, std::max(centile, 0.0)) / 100.0;
    
    double sum = 0.0;
    int i = 0;
    
    for (auto it = input.begin(); it != input.end(); it++, i++)
    {
      sum += *it;
      if (sum > target)
        return static_cast<double>(i - ((sum - target) / *it));
    }
    
    return static_cast<double>(input.size() - 1);
  }
};
        
};  // namespace descriptors
};  // namespace fluid

