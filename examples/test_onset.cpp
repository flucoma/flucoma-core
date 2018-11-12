#include "util/audiofile.hpp"

#include <algorithms/public/OnsetSegmentation.hpp>
#include <algorithms/public/Windows.hpp>
#include <data/FluidTensor.hpp>
#include <algorithms/public/STFT.hpp>

using fluid::FluidTensor;
using fluid::FluidTensorView;
using fluid::audiofile::AudioFileData;
using fluid::audiofile::readFile;
using fluid::audiofile::writeFile;
using fluid::algorithm::OnsetSegmentation;
using fluid::algorithm::STFT;
using fluid::algorithm::windowFuncs;
using fluid::algorithm::WindowType;

using Eigen::ArrayXXd;
using std::ofstream;

int main(int argc, char *argv[]) {
  using std::cout;
  using std::vector;

  if (argc <= 3) {
    cout << "usage: test_onset in.wav filterSize threshold\n";
    return 1;
  }

  AudioFileData data = readFile(argv[1]);
  int nBins = 513;
  int fftSize = 2 * (nBins - 1);
  int hopSize = 256;
  int frameDelta = 512;
  int windowSize = 1024;

  int filterSize = std::stoi(argv[2]);
  double threshold = std::stod(argv[3]);
  STFT stft(windowSize, fftSize, hopSize);
  OnsetSegmentation os(
      fftSize, windowSize, hopSize, frameDelta, WindowType::kHann, threshold,
      OnsetSegmentation::DifferenceFunction::kL1Norm, filterSize);

  fluid::FluidTensor<double, 1> in(data.audio[0]);
  int nFrames = floor((in.size() + hopSize) / hopSize);

  // allocation
  fluid::FluidTensor<std::complex<double>, 2> spec(nFrames, nBins);
  fluid::FluidTensor<double, 2> mag(nFrames, nBins);
  FluidTensor<double, 1> out(os.nFrames(in.size()));

  // processing
  stft.process(in, spec);
  STFT::magnitude(spec, mag);

  os.process(in, out);
  for (int i = 0; i < out.size(); i++) {
    if (out[i] == 1) {
      std::cout << i << std::endl;
    }
  }
  return 0;
}
