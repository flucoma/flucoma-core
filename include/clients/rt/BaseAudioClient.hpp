/***!
 @file fluid::audio::BaseAudioClient
 
 Provides buffering services, and performs simple pass through (i.e. is a concrete class).
 
 Whilst the buffering classes can do overlap and stuff, we're not set up for these here yet.
 
 Override proccess in derived classes to implement algorithms
 
 ***/

#pragma once

#include "data/FluidTensor.hpp"
#include "data/FluidBuffers.hpp"
#include "clients/common/FluidParams.hpp"

namespace fluid {
namespace audio {

    
    template <typename T, typename U>
    class BaseAudioClient
    {
        //Type aliases, mostly for aesthetics
        using source_buffer_type = fluid::FluidSource<T>;
        using sink_buffer_type = fluid::FluidSink<T>;
        using tensor_type = fluid::FluidTensor<T,2>; 
        using view_type = fluid::FluidTensorView<T,2>;
        using VectorView    = fluid::FluidTensorView<T,1>; 
        using const_view_type = const fluid::FluidTensorView<T,2>;
    public:
        
        template <typename V=U>
        struct Signal
        {
        public:
            virtual ~Signal(){}
            virtual void set(V*,V) = 0;
            virtual V& next() = 0;
            virtual void copy_from(VectorView dst, size_t src_offset, size_t size)=0;
            virtual void copy_to(VectorView src, size_t dst_offset, size_t size)=0;
        };
        
        
        
        class AudioSignal: public Signal<U>
        {
        public:
            AudioSignal(){}
            AudioSignal(U* ptr,U elem):m_sig(ptr){}
        
            void set(U* p ,U) override {m_sig = p;}
            
            U& next() override
            {
                return *m_sig++;
            }
            
            void copy_from(VectorView dst, size_t src_offset, size_t size) override
            {
                std::copy(m_sig + src_offset, m_sig + src_offset + size, dst.begin());
            }
            
            void copy_to(VectorView src, size_t dst_offset, size_t size) override
            {
              std::copy(src.begin(),src.end(), m_sig + dst_offset);
            }
            
        private:
            U* m_sig;
        };
        
        
        class ScalarSignal:public Signal<U>
        {
        public:
            ScalarSignal(){};
            ScalarSignal(U* ptr ,U val):m_elem(val){}
            
            void set(U*, U p) override {m_elem = p;}
            
            U& next() override
            {
                return m_elem;
            }
            
            virtual void copy_from(VectorView dst, size_t src_offset, size_t size) override
            {
                std::fill(dst.begin(), dst.end(), m_elem);
            }
            
            virtual void copy_to(VectorView src, size_t dst_offset, size_t size) override
            {
                m_elem = *(src.begin());
            }
            
        
        private:
            U m_elem;
        };
        
        
        
        //No default construction
        BaseAudioClient()= delete;
        //No copying
        BaseAudioClient(BaseAudioClient&)= delete;
//        BaseAudioClient& operator=(BaseAudioClient&)= delete;
        //Default destuctor
        virtual ~BaseAudioClient()=default;
        
        /**!
         New instance taking maximum frame size and the number of channels
         
         You *must* set host buffer size and call reset before attemping to use
         **/
        BaseAudioClient(size_t max_frame_size, size_t n_channels_in = 1, size_t n_channels_out = 1, size_t nIntermediateChannels = 0):
            m_max_frame_size(max_frame_size),
            m_channels_in(n_channels_in),
            m_channels_out(n_channels_out),
            mIntermediateChannels(nIntermediateChannels ? nIntermediateChannels : n_channels_out),
            m_frame(0,0),
            m_frame_out(0,0),
            m_frame_post(0,0),
            m_source(max_frame_size,n_channels_in),
            m_sink(max_frame_size, mIntermediateChannels)
        {
//          newParamSet();
        }
      
      static std::vector<parameter::Descriptor>& getParamDescriptors()
      {
        static std::vector<parameter::Descriptor> descriptors;
        if(descriptors.size() == 0)
        {
          descriptors.emplace_back("winsize", "Window Size", parameter::Type::Long);
          descriptors.back().setMin(4).setDefault(1024).setInstantiation(true);
          
          descriptors.emplace_back("hopsize", "Hop Size", parameter::Type::Long);
          descriptors.back().setMin(1).setDefault(512);
        }
        return descriptors;
      }
      
      
        /**
         TODO: This works for Max /PD, but wouldn't for SC. Come up with something SCish
         
         - Pushes a buffer from the host into our source buffer
         - Pulls some frames out from the source and processes them
         - Pushes these to the sink buffer
         – Reads back to host buffer
         
         Don't override this? Maybe? Doesn't seem like a good idea anyway
         
         Do override process() though. That's what it's for
         **/
        template <typename InputIt, typename OutputIt>
        void do_process(InputIt input,InputIt inend, OutputIt output, OutputIt outend, size_t nsamps, size_t channels_in, size_t channels_out)
        {
            assert(channels_in == m_channels_in);
            assert(channels_out == m_channels_out);
          
          
          
            m_source.push(input,inend,nsamps, channels_in);
            
            //I had imagined we could delegate knowing about the time into the frame
            //to the buffers, but for cases where chunk_size % host_buffer_size !=0
            //we don't call the same number of times each tick
            
            //When we come to worry about overlap, and variable delay times
            // (a) (for overlap) m_max_frame_size size in this look will need to change to take a variable, from somewhere (representing the hop size for this frame _start_)
            // (b) (for varying frame size) the num rows of the view passed in
            // will need to change.
          
            for(; m_frame_time < m_host_buffer_size; m_frame_time+=mHopSize)
            {
                m_source.pull(m_frame,m_frame_time);
                process(m_frame, m_frame_out);
                m_sink.push(m_frame_out,m_frame_time);
            }

            m_frame_time = m_frame_time < m_host_buffer_size?
                m_frame_time : m_frame_time - m_host_buffer_size;
          m_sink.pull(m_frame_post);

          post_process(m_frame_post);

          for(size_t i = 0; (i < m_channels_out && output != outend); ++i,++output)
          {
            (*output)->copy_to(m_frame_post.row(i),0,nsamps);
          }
        }
      
        /**
         Base procesisng method. A no-op in this case
         **/
      virtual void process(view_type in, view_type out) {}
      virtual void post_process(view_type output) {}
      
        
        /**
         Sets the host buffer size. Yes we do need to know this
         
         Call this from host DSP setup
         **/
        void set_host_buffer_size(const size_t size){
            m_host_buffer_size = size;
            m_source.set_host_buffer_size(size);
            m_sink.set_host_buffer_size(size);
            m_frame_post.resize(mIntermediateChannels,m_host_buffer_size);
        }
      
        /**
         Reset everything. Call this from host dsp setup
         **/
      
      std::tuple<bool, std::string> sanityCheck()
      {
        size_t winsize = parameter::lookupParam("winsize", getParams()).getLong();
        
        if(winsize > m_max_frame_size)
        {
          return {false, "Window size out of range"};
        }
        
        return {true, "All is nice"}; 
        
      }
      
      
        virtual void reset()
        {
            m_frame_time = 0;
            m_source.reset();
            m_sink.reset();
          
          size_t windowSize = parameter::lookupParam("winsize", getParams()).getLong();
          mHopSize =  parameter::lookupParam("hopsize", getParams()).getLong();
          
            if(windowSize != m_frame.cols())
            {
              m_frame = FluidTensor<T, 2>(m_channels_in,windowSize);
              m_frame_out = FluidTensor<T, 2>(mIntermediateChannels,windowSize);
            }
        }
        
        size_t channelsOut()
        {
            return m_channels_out;
        }
        
        size_t channelsIn()
        {
            return m_channels_in; 
        }
      
      
      virtual std::vector<parameter::Instance>& getParams() = 0;
     

      
        
    private:
//      void newParamSet()
//      {
//        mParams.clear();
//        for(auto&& d: getParamDescriptors())
//          mParams.emplace_back(d);
//      }
        size_t m_host_buffer_size;
        size_t m_max_frame_size;

        size_t m_frame_time;
        size_t m_channels_in;
        size_t m_channels_out;
        size_t mIntermediateChannels;
      
      size_t mHopSize;
      
        tensor_type m_frame;
        tensor_type m_frame_out;
        tensor_type m_frame_post;
        source_buffer_type m_source;
        sink_buffer_type m_sink;
      
//      std::vector<parameter::Instance> mParams;
      
 
    };
}
}
