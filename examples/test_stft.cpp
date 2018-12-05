#include "util/audiofile.hpp"

#include <algorithms/public/STFT.hpp>
#include <data/FluidTensor.hpp>

int main(int argc, char *argv[]) {
  using fluid::audiofile::AudioFileData;
  using fluid::audiofile::readFile;
  using fluid::audiofile::writeFile;
  using std::cout;
  using std::vector;

  using fluid::algorithm::ISTFT;
  using fluid::algorithm::STFT;

  using fluid::FluidTensor;
  using fluid::FluidTensorView;
  using fluid::Slice;

  if (argc <= 2) {
    cout << "usage: test_stft in.wav out.wav\n";
    return 1;
  }

  AudioFileData data = readFile(argv[1]);
  int fftSize = 2048;
  int windowSize = 1024;
  int hopSize = 128;
  int nBins = fftSize / 2 + 1;
  STFT stft(windowSize, fftSize, hopSize);
  ISTFT istft(windowSize, fftSize, hopSize);
  FluidTensor<double, 1> in(data.audio[0]);
  int nFrames = floor((in.size() + hopSize) / hopSize);
  FluidTensor<std::complex<double>, 2> temp (nFrames, nBins);
  FluidTensor<double, 1> out(data.audio[0]);
  stft.process(in, temp);
  istft.process(temp, out);
  double err = 0;
  for (int i = 0; i < in.size(); i++) {
    //std::cout << "in " << in[i] << std::endl;
    //std::cout << "out " << out[i] << std::endl;
    //std::cout << "err " << std::abs(in[i] - out[i]) << std::endl;
    err += std::abs(in[i] - out[i]);
  }
  data.audio[0] = vector<double>(out.data(), out.data() + out.size());
  writeFile(data, argv[2]);
  std::cout << "----" << std::endl;
  std::cout << "err " << err << std::endl;
  std::cout << "----" << std::endl;
  std::cout << "----" << std::endl;

  // test processFrame
  int frameSize = fftSize / 2 + 1;
  FluidTensorView<double, 1> inF = in(Slice(0, windowSize));
  FluidTensor<double, 1> outF = FluidTensor<double, 1>(windowSize);
  FluidTensor<std::complex<double>, 1> tempF = FluidTensor<std::complex<double>, 1>(frameSize);
  stft.processFrame(inF, tempF);
  istft.processFrame(tempF, outF);
  err = 0;

  FluidTensorView<double, 1> windowNorm = stft.window();
  windowNorm.apply(stft.window(), [](double &x, double &y) { x *= y; });
  windowNorm(0) = 1;

  for (int i = 0; i < inF.size(); i++) {
    //std::cout << "in " << inF[i] << std::endl;
    //std::cout << "out " << outF[i] / windowNorm[i] << std::endl;
    //std::cout << "err " << std::abs(inF[i] - outF[i] / windowNorm[i])<< std::endl;
    err += std::abs(inF[i] - outF[i] / windowNorm[i]);
  }
  std::cout << "----" << std::endl;
  std::cout << "err " << err << std::endl;
  return 0;

}
