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

using fluid::windows::windowFuncs;
using fluid::windows::WindowType;

using Eigen::ArrayXXd;
using std::ofstream;

int main(int argc, char *argv[]) {
  using std::cout;
  using std::vector;

  if (argc <= 1) {
    cout << "usage: test_onset in.wav\n";
    return 1;
  }

  AudioFileData data = readFile(argv[1]);
  int nBins = 513;
  int fftSize = 2 * (nBins - 1);
  int hopSize = 256;
  int frameDelta = 512;
  int windowSize = 1024;
  RealVector in(data.audio[0]);
  RealVectorView inV = RealVectorView(in);

  OnsetSegmentation os(fftSize, windowSize, hopSize, frameDelta,
                    WindowType::Hann, 0.01, OnsetSegmentation::DifferenceFunction::kL1Norm, 20);
  RealVector result(os.nFrames(in.size()));
  RealVectorView outV = RealVectorView(result);
  os.process(inV, outV);
  std::cout<<result<<std::endl;
  return 0;
}
