
#define EIGEN_USE_BLAS

#include <random>
#include <vector>
#include <cmath>

#include <algorithms/Descriptors.hpp>

using fluid::descriptors::Descriptors;
using Real = fluid::FluidTensor<double, 1>;

// Value

void levelDescriptors(Real& input)
{
  double rms = Descriptors::RMS(input);
  double crest = Descriptors::crest(input);
  
  std::cout << "Descriptor RMS " << rms << "\n";
  std::cout << "Descriptor Crest " << crest << "\n";
}

void peakDescriptor(Real& input, double amp)
{
  std::cout << "Amplitude Multiplier Set To  " << amp << "\n";

  Real copy(input.size());
  
  for (auto i = 0; i < input.size(); i++)
    copy(i) = input(i) * amp;
  
  double peak = Descriptors::peak(copy);
  
  std::cout << "Descriptor Peak " << peak << "\n";
}
      
// Main

int main(int argc, const char * argv[])
{
  unsigned int seed = std::random_device()();
  std::mt19937_64 randomGenerator(seed);
  std::normal_distribution<double> gaussian(0.0, 2.0);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  
  int blockSize = 20000;
  double amp = fabs(uniform(randomGenerator));

  Real sine(blockSize);
  Real flatNoise(blockSize);
  Real gaussNoise(blockSize);

  for (auto i = 0; i < blockSize; i++)
    sine(i) = sin(M_PI * i / 25.5);
  
  for (auto i = 0; i < blockSize; i++)
    flatNoise(i) = uniform(randomGenerator);
  
  for (auto i = 0; i < blockSize; i++)
    gaussNoise(i) = gaussian(randomGenerator);
  
  std::cout << "\n*** Sine ***  " << amp << "\n\n";
  
  levelDescriptors(sine);
  peakDescriptor(sine, amp);
  
  std::cout << "\n*** Flat Noise ***  " << amp << "\n\n";
  
  levelDescriptors(flatNoise);
  peakDescriptor(flatNoise, amp);
  
  std::cout << "\n*** Gauss Noise (dev 2.0) ***  " << amp << "\n\n";
  
  levelDescriptors(gaussNoise);
  peakDescriptor(gaussNoise, amp);

  return 0;
}
