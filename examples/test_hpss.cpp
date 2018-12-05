#include "util/audiofile.hpp"

#include <algorithms/public/HPSS.hpp>
#include <algorithms/public/RatioMask.hpp>
#include <algorithms/public/STFT.hpp>
#include <data/FluidTensor.hpp>

#include <fstream>

using fluid::audiofile::AudioFileData;
using fluid::audiofile::readFile;
using fluid::audiofile::writeFile;

using fluid::algorithm::HPSS;
using fluid::algorithm::ISTFT;
using fluid::algorithm::STFT;

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
  fluid::FluidTensor<double, 1> in(data.audio[0]);
  int nFrames = floor((in.size() + hopSize) / hopSize);

  // allocation
  fluid::FluidTensor<std::complex<double>, 2> spec(nFrames, nBins);
  fluid::FluidTensor<double, 2> mag(nFrames, nBins);
  fluid::FluidTensor<double, 2> harmMag(nFrames, nBins);
  fluid::FluidTensor<double, 2> percMag(nFrames, nBins);
  fluid::FluidTensor<double, 2> mixMag(nFrames, nBins);
  fluid::FluidTensor<std::complex<double>, 2> harm(nFrames, nBins);
  fluid::FluidTensor<std::complex<double>, 2> perc(nFrames, nBins);
  fluid::FluidTensor<double, 1> harmonicAudio(in.size());
  fluid::FluidTensor<double, 1> percussiveAudio(in.size());
  // processing
  stft.process(in, spec);
  STFT::magnitude(spec, mag);
  hpsssProcessor.process(mag, harmMag, percMag, mixMag);
  RatioMask mask = RatioMask(mixMag, 1);
  mask.process(spec, harmMag, harm);
  mask.process(spec, percMag, perc);
  istft.process(harm, harmonicAudio);
  istft.process(perc, percussiveAudio);

  data.audio[0] = vector<double>(harmonicAudio.data(),
                                 harmonicAudio.data() + harmonicAudio.size());
  writeFile(data, "harmonic.wav");

  data.audio[0] = vector<double>(
      percussiveAudio.data(), percussiveAudio.data() + percussiveAudio.size());
  writeFile(data, "percussive.wav");

  return 0;
}
