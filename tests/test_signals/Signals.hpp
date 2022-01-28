#include <data/FluidTensor.hpp>

// #include <AudioFile/IAudioFile.h> 

#include <algorithm>
#include <vector>
#include <cmath>

namespace fluid {
namespace testsignals {
  
// constexpr size_t fs = 44100; 

std::vector<double> oneImpulse();
FluidTensor<double,2> stereoImpulses(); 
FluidTensor<double,1> sharpSines(); 
FluidTensor<double, 1> smoothSine();
FluidTensor<double, 2> guitarStrums(const std::string& audio_path);
FluidTensor<double, 2> eurorackSynth(const std::string& audio_path); 

// std::fill(oneImpulse.begin(), oneImpulse.end(), 0);
// oneImpulse[(fs / 2) - 1] = 1.0;

// std::array<double, fs> sharpSines;
// std::generate(sharpSines.begin() + 1000, sharpSines.end(), [i = 1000]() mutable {
//   constexpr double freq = 640;
//   double           sinx = sin(2 * M_PI * i * freq / fs - 1);
// 
//   constexpr double nPeriods = 4;
//   double           phasor = (((fs - 1 - i) % (fs / nPeriods)) / (fs / nPeriods));  
//   i++; 
//   return sinx * phasor;
// });
// 
// std::array<double, fs> smoothSine;
// std::generate(smoothSines.begin(), smoothSines.end(), [i = 0]() mutable
// {
//   double res = sin(2 * M_PI * 320 * i / fs) * fabs(sin(2 * M_PI * i / fs )); 
//   i++; 
//   return res; 
// }); 
// 
// FluidTensor<double,2> impulses(2,fs); 
// impulses.fill(0); 
// 
// impulses.row(0)[1000] = 1; 
// impulses.row(0)[23051] = 1; 
// 
// impulses.row(1)[11030] = 1;
// impules.row(1)[33080] = 1; 


} // namespace testsignals
} // namespace flucoma
