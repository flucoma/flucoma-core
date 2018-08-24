/*
 @file: BaseSTFTClient
 
 Base class for real-time STFT processes
 
*/

#include "clients/rt/BaseAudioClient.hpp"
#include "clients/util/STFTCheckParams.hpp"
#include "algorithms/STFT.hpp"

namespace  fluid {
namespace audio {
    using fluid::stft::STFT;
    using fluid::stft::ISTFT;
    using fluid::stft::STFTCheckParams;
    
    template <typename T, typename U>
    class BaseSTFTClient:public STFTCheckParams, public BaseAudioClient<T,U>
    {
        using data_type = fluid::FluidTensorView<T,2>;
    public:
        BaseSTFTClient() = delete;
        BaseSTFTClient(BaseSTFTClient&) = delete;
        BaseSTFTClient operator=(BaseSTFTClient&) = delete;
        
        BaseSTFTClient(size_t windowsize, size_t hopsize, size_t fftsize):
        STFTCheckParams(windowsize,hopsize,fftsize),
        BaseAudioClient<T,U>(m_window_size,m_hop_size,1,2),
        m_stft(m_window_size, m_fft_size, m_hop_size),
        m_istft(m_window_size, m_fft_size, m_hop_size)
        {
        }
        
        ~BaseSTFTClient() = default;
        
        data_type process(data_type input)
        {
            return m_istft.process_frame(m_stft.process_frame(input.row(0)));
        }
        
    private:
        STFT m_stft;
        ISTFT m_istft;
        
    };
    
    
} //namespace audio
}//namespace fluid

