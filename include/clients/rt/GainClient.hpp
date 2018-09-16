/*!
 @file GainClient.hpp
 
 Simple multi-input client, just does modulation of signal 1 by signal 2, or sclar
 gain change

 */
#ifndef fluid_audio_gainclient_h
#define fluid_audio_gainclient_h

#include "BaseAudioClient.hpp"
#include "clients/common/FluidParams.hpp"

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
        using Signal = typename BaseAudioClient<T,U>::template Signal<U>;
        using AudioSignal = typename BaseAudioClient<T,U>::AudioSignal;
        using ScalarSignal = typename BaseAudioClient<T,U>::ScalarSignal;
      
      static std::vector<parameter::Descriptor> getParamDescriptors()
      {
        static std::vector<parameter::Descriptor> desc;
          
        if(desc.size() == 0)
        {
          
          desc.emplace_back("gain", "Gain", parameter::Type::Float);
          desc.back().setDefault(1);
          BaseAudioClient<T,U>::initParamDescriptors(desc);

        }
        return desc;
      }
      
        /**
         No default instances, no copying
         **/
        GainAudioClient() = delete;
        GainAudioClient(GainAudioClient&) = delete;
        GainAudioClient operator=(GainAudioClient&) = delete;
        
        /**
         Construct with a (maximum) chunk size and some input channels
         **/
        GainAudioClient(size_t max_chunk_size):
        BaseAudioClient<T,U>(max_chunk_size, 2,1)//, output(1,chunk_size) //this has two input channels, one output
        {
          newParamSet();
        }
        
        using  BaseAudioClient<T,U>::channelsIn;
        
        
        /**
         Do the processing: this is the function that descendents of BaseAudioClient should override
         
         Takes a view in
         
         **/
        void process(view_type data, view_type output) override
        {
            //Punishment crashes for the sloppy
            //Data is stored with samples laid out in rows, one channel per row
            assert(output.cols() == data.cols());
            assert(data.rows() == channelsIn());
            
            //Copy the input samples
            output.row(0) = data.row(0);
            
            //Apply gain from the second channel
            output.row(0).apply(data.row(1),[](double& x, double g){
                x *= g;
            });
        }
        
//        /**
//         Having some queriable attribute interface would be longer term goal
//         **/
//        void set_gain(const T gain)
//        {
//            m_scalar_gain = gain;
//        }
      
      void reset() override
      {
        m_scalar_gain = parameter::lookupParam("gain",mParams).getFloat();
        BaseAudioClient<T, U>::reset();
      }
      
      
      std::vector<parameter::Instance>& getParams() override
      {
        return mParams;
      }
      
      
        
    private:
        void newParamSet()
      {
        mParams.clear();
        for(auto&& d: getParamDescriptors())
          mParams.emplace_back(d);
        
      }
      
      
        T m_scalar_gain = 1.;
      std::vector<parameter::Instance> mParams;
      
      
    }; // class
} //namespace audio
} //namespace fluid


#endif /* fluid_audio_gainclient_h */
