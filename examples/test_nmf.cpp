#include "util/audiofile.hpp"
#include <Eigen/Dense>
#include <FluidTensor.hpp>
#include <NMF.hpp>
#include <STFT.hpp>

int main(int argc, char *argv[]) {
  const auto &epsilon = std::numeric_limits<double>::epsilon;

  using std::complex;
  using std::cout;
  using std::vector;

  using Eigen::Map;
  using Eigen::MatrixXcd;
  using Eigen::MatrixXd;

  using fluid::audiofile::AudioFileData;
  using fluid::audiofile::readFile;
  using fluid::audiofile::writeFile;

  using fluid::FluidTensor;
  using fluid::nmf::NMF;
  using fluid::nmf::NMFModel;
  using fluid::stft::ISTFT;
  using fluid::stft::Spectrogram;
  using fluid::stft::STFT;

  using ComplexMatrix = FluidTensor<complex<double>, 2>;
  using RealVector = FluidTensor<double, 1>;
  using MatrixXdMap = Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using MatrixXcdMap = Map<Eigen::Matrix<complex<double>, Eigen::Dynamic,
                                         Eigen::Dynamic, Eigen::RowMajor>>;

  if (argc <= 2) {
    cout << "usage: test_nmf in.wav out.wav\n";
    return 1;
  }

  AudioFileData data = readFile(argv[1]);
  int nBins = 513;
  int fftSize = (2 * nBins) -1;
  int hopSize = 128;
  int rank = 4;
  STFT stft(fftSize, fftSize, hopSize);
  ISTFT istft(fftSize, fftSize, hopSize);
  NMF nmfProcessor(rank, 100);
  RealVector in(data.audio[0]);
  Spectrogram spec = stft.process(in);
  NMFModel decomposition = nmfProcessor.process(spec.getMagnitude());
  MatrixXdMap W(decomposition.W.data(), nBins, rank);
  MatrixXdMap H(decomposition.H.data(), rank, decomposition.H.extent(1));

  MatrixXd source1 = W.col(1) * H.row(1);
  MatrixXd V = W * H;
  MatrixXd filter = source1.cwiseQuotient(V.cwiseMax(epsilon()));
  MatrixXcdMap specMatrix(spec.mData.data(), spec.mData.extent(0),
                          spec.mData.extent(1));
  specMatrix = specMatrix.cwiseProduct(filter.transpose());
  ComplexMatrix out(spec.mData.extent(0), spec.mData.extent(1));
  MatrixXcdMap(out.data(), spec.mData.extent(0), spec.mData.extent(1)) =
      specMatrix;
  Spectrogram resultS(out);
  RealVector audio = istft.process(resultS);
  data.audio[0] = vector<double>(audio.data(), audio.data() + audio.size());
  writeFile(data, argv[2]);

  return 0;
}
