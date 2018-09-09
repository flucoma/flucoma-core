#include "algorithms/RTHPSS.hpp"
#include "algorithms/STFT.hpp"
#include "data/FluidEigenMappings.hpp"
#include "data/FluidTensor.hpp"
#include "util/audiofile.hpp"
#include <fstream>

using fluid::FluidTensor;
using fluid::audiofile::AudioFileData;
using fluid::audiofile::readFile;
using fluid::audiofile::writeFile;
using fluid::eigenmappings::FluidToArrayXXd;
using fluid::rthpss::RTHPSS;
using fluid::stft::ISTFT;
using fluid::stft::Spectrogram;
using fluid::stft::STFT;

using RealMatrix = fluid::FluidTensor<double, 2>;
using RealVector = fluid::FluidTensor<double, 1>;
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
  RTHPSS hpsssProcessor(nBins, vSize, hSize);
  RealVector in(data.audio[0]);
  Spectrogram spec = stft.process(in);
  ComplexMatrix harmonicSpec(spec.mData.rows(), spec.mData.cols());
  ComplexMatrix percussiveSpec(spec.mData.rows(), spec.mData.cols());
  ComplexMatrix result(nBins, 2);
  for (int i = 0; i < spec.mData.rows(); i++) {
    // std::cout<<i<<std::endl;
    hpsssProcessor.processFrame(spec.mData.row(i), result);
    harmonicSpec.row(i) = result.col(0);
    percussiveSpec.row(i) = result.col(1);
  }
  RealVector harmonicAudio = istft.process(harmonicSpec);
  data.audio[0] = vector<double>(harmonicAudio.data(),
                                 harmonicAudio.data() + harmonicAudio.size());
  writeFile(data, "harmonic_rt.wav");
  RealVector percussiveAudio = istft.process(percussiveSpec);
  data.audio[0] = vector<double>(
      percussiveAudio.data(), percussiveAudio.data() + percussiveAudio.size());
  writeFile(data, "percussive_rt.wav");
  return 0;
}
