#include "util/audiofile.hpp"
#include <STFT.hpp>
#include <FluidTensor.hpp>

int main(int argc, char* argv[])
{
  using std::cout;
  using fluid::audiofile::AudioFileData;
  using fluid::audiofile::readFile;
  using fluid::audiofile::writeFile;

  using fluid::stft::STFT;
  using fluid::stft::ISTFT;
  using fluid::stft::Spectrogram;

  if (argc <= 2){
    cout << "usage: test_stft in.wav out.wav\n";
    return 1;
  }

  AudioFileData data = readFile(argv[1]);
  STFT stft (1024, 1024, 128);
  ISTFT istft (1024, 1024, 128);
  fluid::FluidTensor<double, 1> in(data.audio[0]);
  Spectrogram spec = stft.process(in);
  fluid::FluidTensor<double, 1> out = istft.process(spec);
  double err = 0;
  for(int i=0;i<in.size();i++){
    std::cout<<"in "<<in(i)<<std::endl;
    std::cout<<"out "<<out(i)<<std::endl;
    std::cout<<"err "<<std::abs(in(i) - out(i))<<std::endl;
    err+= std::abs(in(i) - out(i));
  }
  data.audio[0] = vector<double>(out.data(), out.data()+out.size());
  writeFile(data,argv[2]);
  std::cout<<"----"<<std::endl;
  std::cout<<"err "<<err<<std::endl;
  return 0;
}
