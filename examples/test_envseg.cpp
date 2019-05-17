#include <HISSTools_AudioFile/BaseAudioFile.h>
#include <HISSTools_AudioFile/IAudioFile.h>
#include <HISSTools_AudioFile/OAudioFile.h>
#include <algorithms/public/EnvelopeSegmentation.hpp>
#include <data/FluidTensor.hpp>

using fluid::FluidTensor;
using fluid::algorithm::EnvelopeSegmentation;

using RealMatrix = FluidTensor<double, 2>;
using RealVector = FluidTensor<double, 1>;

using Eigen::ArrayXXd;
using std::ofstream;

int main(int argc, char *argv[]) {
  using std::cout;
  using std::vector;

  if (argc <= 18) {
    cout << "usage: test_envseg in.wav ... (18 parameters)\n";
    return 1;
  }

  const char *inputPath = argv[1];
  double hiPassFreq = std::stod(argv[2]);
  double rampUpTime = std::stod(argv[3]);
  double rampUpTime2 = std::stod(argv[4]);
  double rampDownTime = std::stod(argv[5]);
  double rampDownTime2 = std::stod(argv[6]);
  double onThreshold = std::stod(argv[7]);
  double relOnThreshold = std::stod(argv[8]);
  double relOffThreshold = std::stod(argv[9]);
  double minTimeAboveThreshold = std::stod(argv[10]);
  double minEventDuration = std::stod(argv[11]);
  double upwardLookupTime = std::stod(argv[12]);
  double offThreshold = std::stod(argv[13]);
  double minTimeBelowThreshold = std::stod(argv[14]);
  double minSilenceDuration = std::stod(argv[15]);
  double downwardLookupTime = std::stod(argv[16]);

  double maxLatency = std::stod(argv[17]);
  int outputType = std::stoi(argv[18]);

  HISSTools::IAudioFile file(inputPath);

  if (!file.isOpen()) {
    std::cout << "File could not be opened\n";
    return -2;
  }

  int frames = file.getFrames();
  auto samplingRate = file.getSamplingRate();

  EnvelopeSegmentation es(std::round(maxLatency * samplingRate), outputType);

  es.init(
          hiPassFreq / samplingRate,                        // hz to fraction
          rampUpTime,
          rampUpTime2,
          rampDownTime,
          rampDownTime2,
          onThreshold,
          relOnThreshold,                                      // dB
          relOffThreshold,                                      // dB
          std::round(minTimeAboveThreshold * samplingRate), // secs to samples
          std::round(minEventDuration * samplingRate),      // secs to samples
          std::round(upwardLookupTime * samplingRate),      // secs to samples
          1*offThreshold,
          std::round(minTimeBelowThreshold * samplingRate), // secs to samples
          std::round(minSilenceDuration * samplingRate),    // secs to samples
          std::round(downwardLookupTime * samplingRate));   // secs to samples

  fluid::FluidTensor<double, 1> in(frames);
  file.readChannel(in.data(), frames, 0);
  fluid::FluidTensor<double, 1> out(frames);
  for (int i = 0; i < in.size(); i++) {
    //std::cout << i << std::endl;
    es.processSample(in(fluid::Slice(i, 1)), out(fluid::Slice(i, 1)));
  }

  HISSTools::OAudioFile outFile(
      "out.wav", HISSTools::BaseAudioFile::kAudioFileWAVE,
      HISSTools::BaseAudioFile::kAudioFileFloat32, 1, samplingRate);
  outFile.writeChannel(out.data(), out.size(), 0);
  return 0;
}
