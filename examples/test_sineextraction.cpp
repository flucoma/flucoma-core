#include "util/audiofile.hpp"

#include <Eigen/Dense>
#include <HISSTools_FFT/HISSTools_FFT.h>

#include <algorithms/public/RatioMask.hpp>
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
  using fluid::algorithm::STFT;
  using fluid::algorithm::RatioMask;

  using fluid::algorithm::SineExtraction;

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
  fluid::FluidTensor<double, 1> in(data.audio[0]);
  int nFrames = floor((in.size() + hopSize) / hopSize);
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
  SineExtraction se(winSize, fftSize, hopSize, 76, 0.7, 15, 0.1, 1.0);

  // allocation
  fluid::FluidTensor<std::complex<double>, 2> spec(nFrames, nBins);
  fluid::FluidTensor<double, 2> mag(nFrames, nBins);
  fluid::FluidTensor<double, 2> sinesMag(nFrames, nBins);
  fluid::FluidTensor<double, 2> noiseMag(nFrames, nBins);
  fluid::FluidTensor<double, 2> mixMag(nFrames, nBins);
  fluid::FluidTensor<std::complex<double>, 2> sines(nFrames, nBins);
  fluid::FluidTensor<std::complex<double>, 2> noise(nFrames, nBins);
  fluid::FluidTensor<double, 1> sinesAudio(in.size());
  fluid::FluidTensor<double, 1> noiseAudio(in.size());

  stft.process(in, spec);
  STFT::magnitude(spec, mag);

  se.process(mag, sinesMag, noiseMag, mixMag);
  RatioMask mask = RatioMask(mixMag, 1);
  mask.process(spec, sinesMag, sines);
  mask.process(spec, noiseMag, noise);
  istft.process(sines, sinesAudio);
  istft.process(noise, noiseAudio);

  data.audio[0] =
      vector<double>(sinesAudio.data(), sinesAudio.data() + sinesAudio.size());
  writeFile(data, "sines.wav");

  data.audio[0] =
      vector<double>(noiseAudio.data(), noiseAudio.data() + noiseAudio.size());
  writeFile(data, "noise.wav");

  return 0;
}
