#include "algorithms/NoveltySegmentation.hpp"
#include "algorithms/STFT.hpp"
#include "data/FluidTensor.hpp"
#include "util/audiofile.hpp"

using fluid::FluidTensor;
using fluid::audiofile::AudioFileData;
using fluid::audiofile::readFile;
using fluid::audiofile::writeFile;
using fluid::stft::Spectrogram;
using fluid::stft::STFT;
using fluid::novelty::NoveltySegmentation;

using RealMatrix = FluidTensor<double, 2>;
using RealVector = FluidTensor<double, 1>;

using Eigen::ArrayXXd;
using std::ofstream;

int main(int argc, char *argv[]) {
  using std::cout;
  using std::vector;

  if (argc <= 2) {
    cout << "usage: test_novelty in.wav kernel_size\n";
    return 1;
  }

  AudioFileData data = readFile(argv[1]);
  int nBins = 513;
  int fftSize = 2 * (nBins - 1);
  int hopSize = 256;
  int kernelSize = std::stoi(argv[2]);
  int windowSize = 1024;
  STFT stft(windowSize, fftSize, hopSize);
  RealVector in(data.audio[0]);
  Spectrogram spec = stft.process(in);
  RealMatrix mag = spec.getMagnitude();
  RealVector result(spec.nFrames());
  NoveltySegmentation nov(kernelSize, 0.8);
  nov.process(mag, result);
  for(int i = 0; i < spec.nFrames(); i++){
    if(result[i] == 1){
      std::cout<<i<<std::endl;
    }
  }
  //std::cout<<result<<std::endl;
  return 0;
}
