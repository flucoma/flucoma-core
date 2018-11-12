#include "util/audiofile.hpp"

#include <algorithms/public/NoveltySegmentation.hpp>
#include <algorithms/public/STFT.hpp>
#include <data/FluidTensor.hpp>

using fluid::FluidTensor;
using fluid::algorithm::NoveltySegmentation;
using fluid::algorithm::STFT;
using fluid::audiofile::AudioFileData;
using fluid::audiofile::readFile;
using fluid::audiofile::writeFile;

using RealMatrix = FluidTensor<double, 2>;
using RealVector = FluidTensor<double, 1>;

using Eigen::ArrayXXd;
using std::ofstream;

int main(int argc, char *argv[]) {
  using std::cout;
  using std::vector;

  if (argc <= 4) {
    cout << "usage: test_novelty in.wav kernel_size filter_size threshold\n";
    return 1;
  }

  AudioFileData data = readFile(argv[1]);
  int nBins = 513;
  int fftSize = 2 * (nBins - 1);
  int hopSize = 256;
  int kernelSize = std::stoi(argv[2]);
  int filterSize = std::stoi(argv[3]);
  double threshold = std::stod(argv[4]);
  int windowSize = 1024;


  STFT stft(windowSize, fftSize, hopSize);
  fluid::FluidTensor<double, 1> in(data.audio[0]);
  int nFrames = floor((in.size() + hopSize) / hopSize);

  // allocation
  fluid::FluidTensor<std::complex<double>, 2> spec(nFrames, nBins);
  fluid::FluidTensor<double, 2> mag(nFrames, nBins);
  FluidTensor<double, 1> novelty(nFrames);

  // processing
  stft.process(in, spec);
  STFT::magnitude(spec, mag);

  NoveltySegmentation nov(kernelSize, threshold, filterSize);
  nov.process(mag, novelty);
  for (int i = 0; i < nFrames; i++) {
    if (novelty[i] == 1) {
      std::cout << i << std::endl;
    }
  }
  // std::cout<<result<<std::endl;
  return 0;
}
