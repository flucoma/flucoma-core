#include "util/audiofile.hpp"

#include <Eigen/Dense>
#include <HISSTools_FFT/HISSTools_FFT.h>

#include <algorithms/public/RTSineExtraction.hpp>
#include <algorithms/public/STFT.hpp>
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
  using fluid::algorithm::STFT;

  using fluid::algorithm::RTSineExtraction;

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
  RTSineExtraction rtse(winSize, fftSize, hopSize, 76, 0.7, 15, 0.1, 1.0);
  fluid::FluidTensor<double, 1> in(data.audio[0]);
  int nFrames = floor((in.size() + hopSize) / hopSize);
    // allocation
  fluid::FluidTensor<std::complex<double>, 2> spec(nFrames, nBins);
  fluid::FluidTensor<double, 2> mag(nFrames, nBins);
  fluid::FluidTensor<std::complex<double>, 2> sines(nFrames, nBins);
  fluid::FluidTensor<std::complex<double>, 2> noise(nFrames, nBins);
  fluid::FluidTensor<double, 1> sinesAudio(in.size());
  fluid::FluidTensor<double, 1> noiseAudio(in.size());

  //processing
  stft.process(in, spec);
  STFT::magnitude(spec, mag);
  fluid::FluidTensor<std::complex<double>, 2> result(nBins, 2);
  for (int i = 0; i < spec.rows(); i++) {
    rtse.processFrame(spec.row(i), result);
    sines.row(i) = result.col(0);
    noise.row(i) = result.col(1);
  }
  istft.process(sines, sinesAudio);
  data.audio[0] =
      vector<double>(sinesAudio.data(), sinesAudio.data() + sinesAudio.size());
  writeFile(data, "sines_rt.wav");
  istft.process(noise, noiseAudio);
  data.audio[0] =
      vector<double>(noiseAudio.data(), noiseAudio.data() + noiseAudio.size());
  writeFile(data, "noise_rt.wav");
  return 0;
}
