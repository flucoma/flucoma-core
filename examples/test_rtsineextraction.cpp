#include <Eigen/Dense>
#include "util/audiofile.hpp"
#include "algorithms/ConvolutionTools.hpp"
#include "data/FluidTensor.hpp"
#include "HISSTools_FFT/HISSTools_FFT.h"
#include "algorithms/STFT.hpp"
#include "algorithms/RTSineExtraction.hpp"

int main(int argc, char *argv[]) {
  using std::complex;
  using std::cout;
  using std::endl;
  using std::ofstream;
  using std::vector;

  using fluid::audiofile::AudioFileData;
  using fluid::audiofile::readFile;
  using fluid::audiofile::writeFile;

  using fluid::FluidTensor;
  using fluid::stft::ISTFT;
  using fluid::stft::Spectrogram;
  using fluid::stft::STFT;

  using Eigen::ArrayXXcd;
  using Eigen::ArrayXXd;
  using Eigen::Map;
  using ArrayXXcdMap = Map<Eigen::Array<complex<double>, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>;
  using ArrayXXdMap = Map<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using ComplexMatrix = FluidTensor<std::complex<double>, 2>;
  using RealMatrix = FluidTensor<double, 2>;
  using RealVector = FluidTensor<double, 1>;
  using fluid::eigenmappings::ArrayXXcdToFluid;
  using fluid::rtsineextraction::RTSineExtraction;

  const auto &epsilon = std::numeric_limits<double>::epsilon;

  if (argc != 2) {
    cout << "usage: test_rtsinemodel in.wav\n";
    return 1;
  }

  AudioFileData data = readFile(argv[1]);

  // Note: this method relies on large fft
  // this parameter may be better hidden from user
  int nBins = 4097;
  int fftSize = 2 * (nBins - 1);
  // these are OK to play with
  int winSize = 2048;
  int hopSize = 512;

  STFT stft(winSize, fftSize, hopSize);
  ISTFT istft(winSize, fftSize, hopSize);
  RealVector in(data.audio[0]);
  Spectrogram spec = stft.process(in);
  // parameters after hop size are:
  // * bandwidth - width in bins of the fragment of window transform correlated with each frame
  // should have an effect on cost vs quality
  // * threshold (0 to 1) select more or less peaks as sinusoidal from the normalized cross-correlation
  // * min length (frames): minimum length of a sinusoidal track (0 for no tracking)
  // * weight of spectral magnitude when associating a peak to an existing track (relative, but suggested 0 to 1)
  // * weight of frequency when associating a peak to an existing track (relativer, suggested 0 to 1)

  RTSineExtraction rtse(winSize, fftSize, hopSize, 76, 0.7, 15, 0.1, 1.0);

  ComplexMatrix sinesSpec(spec.mData.rows(), spec.mData.cols());
  ComplexMatrix noiseSpec(spec.mData.rows(), spec.mData.cols());
  ComplexMatrix result(nBins, 2);
  for (int i = 0; i < spec.mData.rows(); i++) {
    rtse.processFrame(spec.mData.row(i), result);
    sinesSpec.row(i) = result.col(0);
    noiseSpec.row(i) = result.col(1);
  }
  RealVector sinesAudio = istft.process(sinesSpec);
  data.audio[0] = vector<double>(sinesAudio.data(),
                                 sinesAudio.data() + sinesAudio.size());
  writeFile(data, "sines_rt.wav");
  RealVector noiseAudio = istft.process(noiseSpec);
  data.audio[0] = vector<double>(
      noiseAudio.data(), noiseAudio.data() + noiseAudio.size());
  writeFile(data, "noise_rt.wav");

  return 0;
}
