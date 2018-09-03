/**
@file STFTCheckParams.hpp
 
 Util class for checking and constraining STFT arguments
 **/

#include "clients/common/FluidParams.hpp"

#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <cmath> //for log2

namespace fluid {
namespace parameter {

  
  std::tuple<bool, std::string> checkFFTArguments(parameter::Instance& windowSize, parameter::Instance& hopSize, parameter::Instance& fftSize)
  {
    double log2WindowSize = log2(windowSize.getLong());
    if(log2WindowSize - trunc(log2WindowSize) != 0)
    {
      return {false, "Window size must be a power of two"};
    }
    
    //if the FFT size has been changed and is smaller than window size, barf
    //else it defaults to the window size
    if(fftSize.getLong() < windowSize.getLong()){
      if(fftSize.hasChanged())
      {
        return {false, "FFT Size cannot be smaller than window size"};
      }
      else fftSize.setLong(windowSize.getLong());
    }
    //if the FFT size isn't 2^n, barf
    double log2FFTSize = log2(fftSize.getLong());
    if(log2FFTSize - trunc(log2FFTSize) != 0)
    {
      return {false, "FFT size must be a power of two"};
    }
    
    //if the hop size hasn't been changed, it defaults to half the window size
    if(!hopSize.hasChanged())
    {
      hopSize.setLong(windowSize.getLong() / 2);
    }
    
    return {true,""};    
  }
  
  
    class STFTCheckParams
    {
        static constexpr size_t default_fftsize = 2048;
        
    public:
        STFTCheckParams()=delete;
        
        STFTCheckParams(size_t windowsize, size_t hopsize, size_t fft_size):
        m_window_size(windowsize), m_hop_size(hopsize), m_fft_size(fft_size)
        {
            
            std::ostringstream feedback;
            
            if(fft_size < 32)
            {
                
                feedback << fft_size << " is not a sensible FFT Size, setting to default (" << 2048 << ")\n";
                
                m_fft_size = default_fftsize;
                
            }
            else
            {
                double log2_fft_size = log2(m_fft_size);
                if(log2_fft_size - trunc(log2_fft_size) != 0)
                {
                    m_fft_size = pow(2,trunc(log2_fft_size) + 1);
                    feedback <<  "fft_size: " <<  fft_size << "is not a power of two. Increasing fft_size to nearest power of two (" << m_fft_size << ")\n";
                    
                }
            }
            
            if(windowsize > m_fft_size)
            {
                feedback << "Window size can't be greater than fft size: decreasing window size  to " << fft_size <<'\n';
                m_window_size = fft_size;
            }
            
            if(hopsize > m_fft_size / 2)
            {
                feedback << "An overlap factor < 2 may give unpredictable results";
            }
            
            feedback_string.assign(feedback.str());
        }
        
        const std::string getFeedbackString() const
        {
            return feedback_string;
        }
        
        
        size_t m_window_size;
        size_t m_hop_size;
        size_t m_fft_size;
        
    private:
        std::string feedback_string;
        
    };
} //namespace stft
} //namespace fluid
