#include "util/audiofile.hpp"

#include <algorithms/public/RTHPSS.hpp>
#include <algorithms/public/STFT.hpp>
#include <data/FluidEigenMappings.hpp>
#include <data/FluidTensor.hpp>

#include <fstream>

using fluid::FluidTensor;
using fluid::algorithm::FluidToArrayXXd;
using fluid::algorithm::ISTFT;
using fluid::algorithm::RTHPSS;
using fluid::algorithm::Spectrogram;
using fluid::algorithm::STFT;
using fluid::audiofile::AudioFileData;
using fluid::audiofile::readFile;
using fluid::audiofile::writeFile;

using RealMatrix = FluidTensor<double, 2>;
using RealVector = FluidTensor<double, 1>;
using ComplexMatrix = FluidTensor<std::complex<double>, 2>;

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
  RTHPSS hpsssProcessor(nBins, vSize, hSize, 1, 0.2, 0, 0.8, 20, 0.2, 20, 0.8,
                        -20);
  RealVector in(data.audio[0]);
  Spectrogram spec = stft.process(in);
  ComplexMatrix harmonicSpec(spec.mData.rows(), spec.mData.cols());
  ComplexMatrix percussiveSpec(spec.mData.rows(), spec.mData.cols());
  ComplexMatrix residualSpec(spec.mData.rows(), spec.mData.cols());
  ComplexMatrix result(nBins, 3);
  for (int i = 0; i < spec.mData.rows(); i++) {
    hpsssProcessor.processFrame(spec.mData.row(i), result);
    harmonicSpec.row(i) = result.col(0);
    percussiveSpec.row(i) = result.col(1);
    residualSpec.row(i) = result.col(2);
  }
  RealVector harmonicAudio = istft.process(harmonicSpec);
  data.audio[0] = vector<double>(harmonicAudio.data(),
                                 harmonicAudio.data() + harmonicAudio.size());
  writeFile(data, "harmonic_rt.wav");

  RealVector percussiveAudio = istft.process(percussiveSpec);
  data.audio[0] = vector<double>(
      percussiveAudio.data(), percussiveAudio.data() + percussiveAudio.size());
  writeFile(data, "percussive_rt.wav");

  RealVector residualAudio = istft.process(residualSpec);
  data.audio[0] = vector<double>(residualAudio.data(),
                                 residualAudio.data() + residualAudio.size());
  writeFile(data, "residual_rt.wav");

  return 0;
}
