#include "algorithms/OnsetSegmentation.hpp"
#include "algorithms/Windows.hpp"
#include "data/FluidTensor.hpp"
#include "util/audiofile.hpp"

using fluid::FluidTensor;
using fluid::FluidTensorView;
using fluid::audiofile::AudioFileData;
using fluid::audiofile::readFile;
using fluid::audiofile::writeFile;
using fluid::onset::OnsetSegmentation;
using RealMatrix = FluidTensor<double, 2>;
using RealVector = FluidTensor<double, 1>;
using RealVectorView = FluidTensorView<double, 1>;

using fluid::windows::WindowType;
using fluid::windows::windowFuncs;

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
  RealVector in(data.audio[0]);
  int filterSize = std::stoi(argv[2]);
  double threshold = std::stod(argv[3]);

  RealVectorView inV = RealVectorView(in);

  OnsetSegmentation os(
      fftSize, windowSize, hopSize, frameDelta, WindowType::Hann, threshold,
      OnsetSegmentation::DifferenceFunction::kL1Norm, filterSize);
  RealVector result(os.nFrames(in.size()));
  RealVectorView outV = RealVectorView(result);
  os.process(inV, outV);
  for (int i = 0; i < result.size(); i++) {
    if (result[i] == 1) {
      std::cout << i << std::endl;
    }
  }
  return 0;
}
