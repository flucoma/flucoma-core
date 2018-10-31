#include "util/audiofile.hpp"

#include <Eigen/Dense>
#include <HISSTools_FFT/HISSTools_FFT.h>

#include <algorithms/public/STFT.hpp>
#include <algorithms/public/SineExtraction.hpp>
#include <data/FluidTensor.hpp>

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
  using fluid::algorithm::ISTFT;
  using fluid::algorithm::Spectrogram;
  using fluid::algorithm::STFT;

  using Eigen::ArrayXXcd;
  using Eigen::ArrayXXd;
  using Eigen::Map;
  using ArrayXXcdMap = Map<Eigen::Array<complex<double>, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>;
  using ArrayXXdMap = Map<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using fluid::algorithm::SineExtraction;
  using fluid::algorithm::SinesPlusNoiseModel;
  using ComplexMatrix = FluidTensor<complex<double>, 2>;
  using RealMatrix = FluidTensor<double, 2>;
  using RealVector = FluidTensor<double, 1>;

  const auto &epsilon = std::numeric_limits<double>::epsilon;

  if (argc != 2) {
    cout << "usage: test_sinemodel in.wav\n";
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
  // * bandwidth - width in bins of the fragment of window transform correlated
  // with each frame should have an effect on cost vs quality
  // * threshold (0 to 1) select more or less peaks as sinusoidal from the
  // normalized cross-correlation
  // * min length (frames): minimum length of a sinusoidal track (0 for no
  // tracking)
  // * weight of spectral magnitude when associating a peak to an existing track
  // (relative, but suggested 0 to 1)
  // * weight of frequency when associating a peak to an existing track
  // (relativer, suggested 0 to 1)

  SineExtraction se(winSize, fftSize, hopSize, 76, 0.7, 15, 0.1, 1.0);
  RealMatrix mag = spec.getMagnitude();

  SinesPlusNoiseModel result = se.process(mag);

  ArrayXXcdMap specArray(spec.mData.data(), spec.mData.extent(0),
                         spec.mData.extent(1));
  ArrayXXdMap sinesMag(result.sines.data(), result.sines.extent(0),
                       result.sines.extent(1));
  ArrayXXdMap noiseMag(result.noise.data(), result.noise.extent(0),
                       result.noise.extent(1));
  ArrayXXd sum = sinesMag + noiseMag;
  sum = sum.max(epsilon());

  RealMatrix sineMask(spec.mData.extent(0), spec.mData.extent(1));
  ArrayXXdMap(sineMask.data(), spec.mData.extent(0), spec.mData.extent(1)) =
      sinesMag / sum;

  ArrayXXcd sines = specArray * sinesMag / sum;
  ArrayXXcd noise = specArray * noiseMag / sum;

  ComplexMatrix sinesOut(spec.mData.extent(0), spec.mData.extent(1));
  ArrayXXcdMap(sinesOut.data(), spec.mData.extent(0), spec.mData.extent(1)) =
      sines;
  Spectrogram sinesSpec(sinesOut);
  RealMatrix sinesSpecMag = sinesSpec.getMagnitude();
  RealVector sinesAudio = istft.process(sinesSpec);
  data.audio[0] =
      vector<double>(sinesAudio.data(), sinesAudio.data() + sinesAudio.size());
  writeFile(data, "sines.wav");

  ComplexMatrix noiseOut(spec.mData.extent(0), spec.mData.extent(1));
  ArrayXXcdMap(noiseOut.data(), spec.mData.extent(0), spec.mData.extent(1)) =
      noise;
  Spectrogram noiseSpec(noiseOut);
  RealVector noiseAudio = istft.process(noiseSpec);
  data.audio[0] =
      vector<double>(noiseAudio.data(), noiseAudio.data() + noiseAudio.size());
  writeFile(data, "noise.wav");

  return 0;
}
