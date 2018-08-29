#include "util/audiofile.hpp"
#include "algorithms/STFT.hpp"
#include "data/FluidTensor.hpp"

int main(int argc, char* argv[])
{
  using std::cout;
  using std::vector;
  using fluid::audiofile::AudioFileData;
  using fluid::audiofile::readFile;
  using fluid::audiofile::writeFile;

  using fluid::stft::STFT;
  using fluid::stft::ISTFT;
  using fluid::stft::Spectrogram;

  using fluid::FluidTensor;
  using fluid::FluidTensorView;
  using fluid::slice;

  if (argc <= 2){
    cout << "usage: test_stft in.wav out.wav\n";
    return 1;
  }

  AudioFileData data = readFile(argv[1]);
  STFT stft (1024, 2048, 128);
  ISTFT istft (1024, 2048, 128);
  FluidTensor<double, 1> in(data.audio[0]);
  Spectrogram spec = stft.process(in);
  FluidTensor<double, 1> out = istft.process(spec);
  double err = 0;
  for(int i=0;i<in.size();i++){
    std::cout<<"in "<<in[i]<<std::endl;
    std::cout<<"out "<<out[i]<<std::endl;
    std::cout<<"err "<<std::abs(in[i] - out[i])<<std::endl;
    err+= std::abs(in[i] - out[i]);
  }
  data.audio[0] = vector<double>(out.data(), out.data()+out.size());
  writeFile(data,argv[2]);
  std::cout<<"----"<<std::endl;
  std::cout<<"err "<<err<<std::endl;
  std::cout<<"----"<<std::endl;
  std::cout<<"----"<<std::endl;

  // test processFrame
  FluidTensorView<double, 1> inF = in(slice(0, 1024));
  FluidTensorView<double, 2> outF = istft.processFrame(stft.processFrame(inF));
  std::cout<<"size "<<outF.size()<<std::endl;
  err = 0;
  outF(1, 0) = 1;
  for(int i=0;i<inF.size();i++){
    std::cout<<"in "<<inF[i]<<std::endl;

    std::cout<<"out "<<outF(0, i) / outF(1, i)<<std::endl;
    std::cout<<"err "<<std::abs(inF[i] - outF(0, i) / outF(1, i))<<std::endl;
    err += std::abs(inF[i] - outF(0, i) / outF(1, i));
  }
  std::cout<<"----"<<std::endl;
  std::cout<<"err "<<err<<std::endl;
  return 0;
}
