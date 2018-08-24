/*!
 @file GainClient.hpp
 
 Simple multi-input client, just does modulation of signal 1 by signal 2, or sclar
 gain change

 */
#ifndef fluid_audio_gainclient_h
#define fluid_audio_gainclient_h

#include "BaseAudioClient.hpp"

namespace fluid {
namespace audio {
    
    /**!
     @class GainAudioClient
     
     Inherits core functionality (incl. variable hop size input ansd output buffering) from BaseAudioClient<T>
     
     **/
    template <typename T, typename U>
    class GainAudioClient: public BaseAudioClient<T,U>
    {
        using tensor_type = fluid::FluidTensor<T,2>;
        using view_type = fluid::FluidTensorView<T,2>;
    public:
        using signal_type = typename BaseAudioClient<T,U>::template Signal<U>;
        using audio_signal = typename BaseAudioClient<T,U>::AudioSignal;
        using scalar_signal = typename BaseAudioClient<T,U>::ScalarSignal;
        /**
         No default instances, no copying
         **/
        GainAudioClient() = delete;
        GainAudioClient(GainAudioClient&) = delete;
        GainAudioClient operator=(GainAudioClient&) = delete;
        
        /**
         Construct with a (maximum) chunk size and some input channels
         **/
        GainAudioClient(size_t chunk_size,size_t hop_size):
        BaseAudioClient<T,U>(chunk_size,hop_size, 2,1), output(1,chunk_size) //this has two input channels, one output
        {}
        
        using  BaseAudioClient<T,U>::channelsIn;
        
        
        /**
         Do the processing: this is the function that descendents of BaseAudioClient should override
         
         Takes a view in
         
         **/
        virtual view_type process(view_type data)
        {
            //Punishment crashes for the sloppy
            //Data is stored with samples laid out in rows, one channel per row
            assert(output.cols() == data.cols());
            assert(data.rows() == channelsIn());
            
            //Copy the input samples
            output = data.row(0);
            
            //Apply gain from the second channel
            output.row(0).apply(data.row(1),[](double& x, double g){
                x *= g;
            });
            return output;
        }
        
        /**
         Having some queriable attribute interface would be longer term goal
         **/
        void set_gain(const T gain)
        {
            m_scalar_gain = gain; 
        }
        
    private:
        tensor_type output;
        T m_scalar_gain = 1.;
    }; // class
} //namespace audio
} //namespace fluid


#endif /* fluid_audio_gainclient_h */
