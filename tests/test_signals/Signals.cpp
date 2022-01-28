#include "Signals.hpp"
#include <data/FluidTensor.hpp>

#include <AudioFile/IAudioFile.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace fluid {
namespace testsignals {

constexpr size_t fs = 44100;

std::vector<double> oneImpulse()
{
  std::vector<double> oneImpulse(fs);
  std::fill(oneImpulse.begin(), oneImpulse.end(), 0);
  oneImpulse[(fs / 2) - 1] = 1.0;
  return oneImpulse;
}

FluidTensor<double, 2> stereoImpulses()
{
  FluidTensor<double, 2> impulses(2, fs);
  impulses.fill(0);
  // 1000.0, 12025.0, 23051.0, 34076.0
  impulses.row(0)[1000] = 1;
  impulses.row(0)[23051] = 1;

  impulses.row(1)[12025] = 1;
  impulses.row(1)[34076] = 1;
  return impulses;
}

FluidTensor<double, 1> sharpSines()
{
  FluidTensor<double, 1> sharpSines(fs);
  std::generate(sharpSines.begin() + 1000, sharpSines.end(),
                [i = 1000]() mutable {
                  constexpr double freq = 640;
                  double           sinx = sin(2 * M_PI * i * freq / fs - 1);
                  constexpr double nPeriods = 4;
                  double           phasor =
                      (((fs - 1 - i) % index(fs / nPeriods)) / (fs / nPeriods));
                  i++;
                  return sinx * phasor;
                });
  return sharpSines;
}

FluidTensor<double, 1> smoothSine()
{
  FluidTensor<double, 1> smoothSine(fs);
  std::generate(smoothSine.begin(), smoothSine .end(), [i = 0]() mutable {
    double res = sin(2 * M_PI * 320 * i / fs) * fabs(sin(2 * M_PI * i / fs));
    i++;
    return res;
  });
  return smoothSine;
}

FluidTensor<double, 2> load(const std::string& audio_path, const std::string& file)
{
   HISSTools::IAudioFile f(audio_path + "/" + file);
   auto e = f.getErrors();
   if(e.size())
   {
    throw std::runtime_error(HISSTools::BaseAudioFile::getErrorString(e[0]));
   }
   
   FluidTensor<double,2> data(f.getChannels(),f.getFrames());
   f.readInterleaved(data.data(), f.getFrames());
   e = f.getErrors();
   if(e.size())
   {
    throw std::runtime_error(HISSTools::BaseAudioFile::getErrorString(e[0]));
   }
   
   return data;
}

FluidTensor<double, 2> guitarStrums(const std::string& audio_path)
{
  static std::string file{"Tremblay-AaS-AcousticStrums-M.wav"};
  return load(audio_path,file);
}

FluidTensor<double, 2> eurorackSynth(const std::string& audio_path)
{
  static std::string file{"Tremblay-AaS-SynthTwoVoices-M.wav"};
  return load(audio_path,file);
}


} // namespace testsignals
} // namespace fluid
