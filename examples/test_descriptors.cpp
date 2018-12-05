#include <cmath>
#include <random>
#include <vector>

#include <algorithms/util/Descriptors.hpp>
#include <algorithms/util/FFT.hpp>
#include <algorithms/util/Windows.hpp>
#include <algorithms/util/FluidEigenMappings.hpp>

using fluid::descriptors::Descriptors;
using fluid::eigenmappings::FluidToArrayXXd;
using fluid::windows::windowFuncs;
using Real = fluid::FluidTensor<double, 1>;

void calcSpectrum(Real &output, Real &input) {
  using fluid::fft::FFT;

  FFT processor(input.size());

  Eigen::VectorXd mapped(input.size());
  std::vector<double> window =
      windowFuncs[fluid::windows::WindowType::Hann](input.size());

  for (auto i = 0; i < input.size(); i++)
    mapped(i) = input(i) * window[i] * 2.0;

  auto complexSpectrum = processor.process(mapped);

  for (auto i = 0; i < output.size(); i++) {
    const double real = complexSpectrum(i).real();
    const double imag = complexSpectrum(i).imag();
    output(i) = sqrt(real * real + imag * imag);
  }
}

void calcDescriptors(Real &input, Real &spectrum, double amp) {
  double rms = Descriptors::RMS(input);
  double crest = Descriptors::crest(input);

  std::cout << "Descriptor RMS " << rms << "\n";
  std::cout << "Descriptor Crest " << crest << "\n";

  std::cout << "Amplitude Multiplier Set To  " << amp << "\n";

  Real copy(input.size());

  for (auto i = 0; i < input.size(); i++)
    copy(i) = input(i) * amp;

  double peak = Descriptors::peak(copy);

  std::cout << "Descriptor Peak " << peak << "\n";

  double rolloff = Descriptors::rolloff(spectrum);
  double flatness = Descriptors::flatness(spectrum);
  double centroid = Descriptors::centroid(spectrum);
  double spread = Descriptors::spread(spectrum);
  double skewness = Descriptors::skewness(spectrum);
  double kurtosis = Descriptors::kurtosis(spectrum);

  std::cout << "Descriptor Rolloff " << rolloff << "\n";
  std::cout << "Descriptor Flatness " << flatness << "\n";
  std::cout << "Descriptor Centroid " << centroid << "\n";
  std::cout << "Descriptor Spread " << spread << "\n";
  std::cout << "Descriptor Skewness " << skewness << "\n";
  std::cout << "Descriptor Kurtosis " << kurtosis << "\n";
}

// Main

int main(int argc, const char *argv[]) {
  unsigned int seed = std::random_device()();
  std::mt19937_64 randomGenerator(seed);
  std::normal_distribution<double> gaussian(0.0, 2.0);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);

  int blockSize = 8192;
  double amp = fabs(uniform(randomGenerator));

  Real sine(blockSize);
  Real flatNoise(blockSize);
  Real gaussNoise(blockSize);

  Real sineSpectrum((blockSize / 2) + 1);
  Real flatNoiseSpectrum((blockSize / 2) + 1);
  Real gaussNoiseSpectrum((blockSize / 2) + 1);

  for (auto i = 0; i < blockSize; i++)
    sine(i) = sin(M_PI * i / 25.5);

  for (auto i = 0; i < blockSize; i++)
    flatNoise(i) = uniform(randomGenerator);

  for (auto i = 0; i < blockSize; i++)
    gaussNoise(i) = gaussian(randomGenerator);

  calcSpectrum(sineSpectrum, sine);
  calcSpectrum(flatNoiseSpectrum, flatNoise);
  calcSpectrum(gaussNoiseSpectrum, gaussNoise);

  std::cout << "\n*** Sine ***  " << amp << "\n\n";

  calcDescriptors(sine, sineSpectrum, amp);

  std::cout << "\n*** Flat Noise ***  " << amp << "\n\n";

  calcDescriptors(flatNoise, flatNoiseSpectrum, amp);

  std::cout << "\n*** Gauss Noise (dev 2.0) ***  " << amp << "\n\n";

  calcDescriptors(gaussNoise, gaussNoiseSpectrum, amp);

  return 0;
}
