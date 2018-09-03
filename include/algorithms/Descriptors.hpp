
#pragma once

#include "../data/FluidTensor.hpp"

#include <cmath>
#include <algorithm>

namespace fluid {
namespace descriptors {
  
class Descriptors
{
  using Real = fluid::FluidTensorView<double, 1>;

  // Helper structs
  
  struct NOP  { double operator()(double a) { return a; } };
  struct Pow2 { double operator()(double a) { return a * a; } };
  struct Pow3 { double operator()(double a) { return a * a * a; } };
  struct Pow4 { double operator()(double a) { return Pow2()(Pow2()(a)); } };
  struct Log  { double operator()(double a) { return log(a); } };

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
    return statMax(input) / RMS(input);
  }
  
  static double peak(const Real& input)
  {
    return statMax(input);
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
  
  static double statSum(const Real& input)
  {
    return statSum(input, NOP());
  }
  
  static double statSumSquares(const Real& input)
  {
    return statSum(input, Pow2());
  }
  
  static double statSumLogs(const Real& input)
  {
    return statSum(input, Log());
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
  
  static double statMin(const Real& input)
  {
    double min = std::numeric_limits<double>::infinity();
    
    for (auto it = input.begin(); it != input.end(); it++)
      min = std::min(min, *it);
    
    return min;
  }
  
  static double statMax(const Real& input)
  {
    double max = -std::numeric_limits<double>::infinity();
    
    for (auto it = input.begin(); it != input.end(); it++)
      max = std::max(max, *it);

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

