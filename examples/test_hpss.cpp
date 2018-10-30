#include "algorithms/MedianFilter.hpp"

#include "algorithms/HPSS.hpp"
#include "algorithms/RatioMask.hpp"
#include "algorithms/STFT.hpp"
#include "data/FluidEigenMappings.hpp"
#include "data/FluidTensor.hpp"
#include "util/audiofile.hpp"
#include <fstream>

using fluid::algorithm::FluidToArrayXXd;
using fluid::audiofile::AudioFileData;
using fluid::audiofile::readFile;
using fluid::audiofile::writeFile;

using fluid::algorithm::HPSS;
using fluid::algorithm::HPSSModel;
using fluid::algorithm::ISTFT;
using fluid::algorithm::STFT;
using fluid::algorithm::Spectrogram;

using RealMatrix = fluid::FluidTensor<double, 2>;
using RealVector = fluid::FluidTensor<double, 1>;

using Eigen::ArrayXXd;
using fluid::algorithm::RatioMask;
using std::ofstream;

int main(int argc, char *argv[]) {
  using std::cout;
  using std::vector;

  if (argc <= 3) {
    cout << "usage: test_hpss in.wav v_size h_size\n";
    return 1;
  }

  AudioFileData data = readFile(argv[1]);
  int nBins = 1025;
  int fftSize = 2 * (nBins - 1);
  int hopSize = 256;
  int vSize = std::stoi(argv[2]);
  int hSize = std::stoi(argv[3]);
  int windowSize = 2048;
  STFT stft(windowSize, fftSize, hopSize);
  ISTFT istft(windowSize, fftSize, hopSize);
  HPSS hpsssProcessor(vSize, hSize);
  RealVector in(data.audio[0]);
  Spectrogram spec = stft.process(in);
  RealMatrix mag = spec.getMagnitude();

  HPSSModel decomposition = hpsssProcessor.process(mag);
  RatioMask mask = RatioMask(decomposition.getMixEstimate(), 1);

  Spectrogram harmonicSpec =
      mask.process(spec.mData, decomposition.getHarmonicEstimate());
  Spectrogram percussiveSpec =
      mask.process(spec.mData, decomposition.getPercussiveEstimate());

  RealVector harmonicAudio = istft.process(harmonicSpec);
  data.audio[0] = vector<double>(harmonicAudio.data(),
                                 harmonicAudio.data() + harmonicAudio.size());
  writeFile(data, "harmonic.wav");

  RealVector percussiveAudio = istft.process(percussiveSpec);
  data.audio[0] = vector<double>(
      percussiveAudio.data(), percussiveAudio.data() + percussiveAudio.size());
  writeFile(data, "percussive.wav");

  return 0;
}
