#include "util/audiofile.hpp"

#include <algorithms/public/RTHPSS.hpp>
#include <algorithms/public/STFT.hpp>
#include <algorithms/util/FluidEigenMappings.hpp>
#include <data/FluidTensor.hpp>

#include <fstream>

using fluid::FluidTensor;
using fluid::algorithm::ISTFT;
using fluid::algorithm::RTHPSS;
using fluid::algorithm::STFT;
using fluid::audiofile::AudioFileData;
using fluid::audiofile::readFile;
using fluid::audiofile::writeFile;

using Eigen::ArrayXXd;
using std::ofstream;

int main(int argc, char *argv[]) {
  using std::cout;
  using std::vector;

  if (argc <= 3) {
    cout << "usage: test_rthpss in.wav v_size h_size\n";
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
  RTHPSS hpsssProcessor;
  hpsssProcessor.init(nBins, vSize, hSize, vSize, hSize, 0, 0.2, 0, 0.8, 20,
                      0.2, 20, 0.8, -20);
  fluid::FluidTensor<double, 1> in(data.audio[0]);
  int nFrames = floor((in.size() + hopSize) / hopSize);
  fluid::FluidTensor<std::complex<double>, 2> spec(nFrames, nBins);
  fluid::FluidTensor<double, 2> mag(nFrames, nBins);
  fluid::FluidTensor<double, 2> harmMag(nFrames, nBins);
  fluid::FluidTensor<double, 2> percMag(nFrames, nBins);
  fluid::FluidTensor<double, 2> resMag(nFrames, nBins);
  fluid::FluidTensor<double, 2> mixMag(nFrames, nBins);
  fluid::FluidTensor<std::complex<double>, 2> harm(nFrames, nBins);
  fluid::FluidTensor<std::complex<double>, 2> perc(nFrames, nBins);
  fluid::FluidTensor<std::complex<double>, 2> residual(nFrames, nBins);
  fluid::FluidTensor<std::complex<double>, 2> result(nBins, 3);
  fluid::FluidTensor<double, 1> harmonicAudio(in.size());
  fluid::FluidTensor<double, 1> percussiveAudio(in.size());
  fluid::FluidTensor<double, 1> residualAudio(in.size());

  //hpsssProcessor.setHSize(13);
  //hpsssProcessor.setVSize(13);

  stft.process(in, spec);
  for (int i = 0; i < nFrames; i++) {
    hpsssProcessor.processFrame(spec.row(i), result);
    harm.row(i) = result.col(0);
    perc.row(i) = result.col(1);
    residual.row(i) = result.col(2);
  }
  //hpsssProcessor.setHSize(hSize);
  //hpsssProcessor.setVSize(vSize);
  /*
  for (int i = 0; i < nFrames; i++) {
    hpsssProcessor.processFrame(spec.row(i), result);
    harm.row(i) = result.col(0);
    perc.row(i) = result.col(1);
    residual.row(i) = result.col(2);
  }*/

  istft.process(harm, harmonicAudio);
  istft.process(perc, percussiveAudio);
  istft.process(residual, residualAudio);

  data.audio[0] = vector<double>(harmonicAudio.data(),
                                 harmonicAudio.data() + harmonicAudio.size());
  writeFile(data, "harmonic_rt.wav");

  data.audio[0] = vector<double>(
      percussiveAudio.data(), percussiveAudio.data() + percussiveAudio.size());
  writeFile(data, "percussive_rt.wav");

  data.audio[0] = vector<double>(residualAudio.data(),
                                 residualAudio.data() + residualAudio.size());
  writeFile(data, "residual_rt.wav");

  return 0;
}
