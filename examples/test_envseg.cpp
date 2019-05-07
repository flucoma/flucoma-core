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

  if (argc <= 13) {
    cout << "usage: test_envseg in.wav ... (13 parameters)\n";
    return 1;
  }

  const char *inputPath = argv[1];
  double hiPassFreq = std::stod(argv[2]);
  double rampUpTime = std::stod(argv[3]);
  double rampDownTime = std::stod(argv[4]);
  double onThreshold = std::stod(argv[5]);
  double minTimeAboveThreshold = std::stod(argv[6]);
  double minEventDuration = std::stod(argv[7]);
  double upwardLookupTime = std::stod(argv[8]);
  double offThreshold = std::stod(argv[9]);
  double minTimeBelowThreshold = std::stod(argv[10]);
  double minSilenceDuration = std::stod(argv[11]);
  double downwardLookupTime = std::stod(argv[12]);

  double maxLatency = std::stod(argv[13]);
  int outputType = std::stoi(argv[14]);

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
          std::round(rampUpTime * samplingRate),            // secs to samples
          std::round(rampDownTime * samplingRate),          // secs to samples
          onThreshold,                                      // dB
          std::round(minTimeAboveThreshold * samplingRate), // secs to samples
          std::round(minEventDuration * samplingRate),      // secs to samples
          std::round(upwardLookupTime * samplingRate),      // secs to samples
          offThreshold,
          std::round(minTimeBelowThreshold * samplingRate), // secs to samples
          std::round(minSilenceDuration * samplingRate),    // secs to samples
          std::round(downwardLookupTime * samplingRate));   // secs to samples

  fluid::FluidTensor<double, 1> in(frames);
  file.readChannel(in.data(), frames, 0);
  fluid::FluidTensor<double, 1> out(frames);
  for (int i = 0; i < in.size(); i++) {
    es.processSample(in(fluid::Slice(i, 1)), out(fluid::Slice(i, 1)));
  }

  HISSTools::OAudioFile outFile(
      "out.wav", HISSTools::BaseAudioFile::kAudioFileWAVE,
      HISSTools::BaseAudioFile::kAudioFileFloat32, 1, samplingRate);
  outFile.writeChannel(out.data(), out.size(), 0);
  return 0;
}
