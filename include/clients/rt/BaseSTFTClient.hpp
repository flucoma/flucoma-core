/*
 @file: BaseSTFTClient

 Base class for real-time STFT processes

*/

#include "clients/rt/BaseAudioClient.hpp"
#include "clients/common/STFTCheckParams.hpp"
#include "algorithms/STFT.hpp"

namespace  fluid {
namespace audio {


    template <typename T, typename U>
  class BaseSTFTClient:public BaseAudioClient<T,U>
    {
      using data_type = FluidTensorView<T,2>;
      using vector    = FluidTensor<T,1>;
      using complex   = FluidTensorView<std::complex<T>,1>;
      
    public:
        BaseSTFTClient() = delete;
        BaseSTFTClient(BaseSTFTClient&) = delete;
        BaseSTFTClient operator=(BaseSTFTClient&) = delete;

        BaseSTFTClient(size_t windowsize, size_t hopsize, size_t fftsize):
        //stft::STFTCheckParams(windowsize,hopsize,fftsize),
        BaseAudioClient<T,U>(windowsize,hopsize,1,1,2),
        m_window_size(windowsize), m_hop_size(hopsize), m_fft_size(fftsize),
        m_stft(m_window_size, m_fft_size, m_hop_size),
        m_istft(m_window_size, m_fft_size, m_hop_size)
        {
          //Make the product of the input and output windows
          normWindow = m_stft.window();
          normWindow.apply(m_istft.window(),[](double& x, double& y)
          {
            x *= y;
          });
//          std::cout << normWindow;
        }

        ~BaseSTFTClient() = default;

        void process(data_type input, data_type output) override
        {
            complex spec  = m_stft.processFrame(input.row(0));
            output.row(0) = m_istft.processFrame(spec);
            output.row(1) = normWindow;
        }
      
        void post_process(data_type output) override
        {
          
//          std::cout << "OUT\n" << output << '\n';
          
          output.row(0).apply(output.row(1),[](double& x, double g){
              if(x)
              {
                x /= g ? g : 1;
              }
          });
        }

    private:
      size_t m_window_size;
      size_t m_hop_size;
      size_t m_fft_size;
      stft::STFT m_stft;
      stft::ISTFT m_istft;
      vector normWindow;
    };


} //namespace audio
}//namespace fluid
