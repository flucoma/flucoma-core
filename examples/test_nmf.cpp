#include "util/audiofile.hpp"
#include <FluidTensor.hpp>
#include <NMF.hpp>
#include <RatioMask.hpp>
#include <STFT.hpp>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  const auto &epsilon = std::numeric_limits<double>::epsilon;

  using std::complex;
  using std::cout;
  using std::vector;

  using Eigen::Map;
  using Eigen::MatrixXcd;
  using Eigen::MatrixXd;

  using fluid::audiofile::AudioFileData;
  using fluid::audiofile::readFile;
  using fluid::audiofile::writeFile;

  using fluid::FluidTensor;
  using fluid::nmf::NMF;
  using fluid::nmf::NMFModel;
  using fluid::stft::ISTFT;
  using fluid::stft::Spectrogram;
  using fluid::stft::STFT;

  using ComplexMatrix = FluidTensor<complex<double>, 2>;
  using RealVector = FluidTensor<double, 1>;
  using RealMatrix = FluidTensor<double, 2>;

  using fluid::ratiomask::RatioMask;

  if (argc <= 2) {
    cout << "usage: test_nmf in.wav rank\n";
    return 1;
  }

  AudioFileData data = readFile(argv[1]);
  int nBins = 1025;
  int fftSize = 2 * (nBins - 1) ;
  int hopSize = 128;
  int rank = std::stoi(argv[2]);
  int windowSize = 2048;
  STFT stft(windowSize, fftSize, hopSize);
  ISTFT istft(windowSize, fftSize, hopSize);
  NMF nmfProcessor(rank, 100);
  RealVector in(data.audio[0]);
  Spectrogram spec = stft.process(in);
  NMFModel decomposition = nmfProcessor.process(spec.getMagnitude());

  RatioMask mask = RatioMask(decomposition.getMixEstimate(), 1);

  for (int i = 0; i < rank; i++) {
    RealMatrix estimate = decomposition.getEstimate(i);
    Spectrogram result(mask.process(spec.mData, estimate));
    RealVector audio = istft.process(result);
    data.audio[0] = vector<double>(audio.data(), audio.data() + audio.size());
    std::string fname = "source_" + std::to_string(i) + ".wav";
    writeFile(data, fname.c_str());
  }

  return 0;
}
