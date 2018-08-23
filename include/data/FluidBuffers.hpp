/*!
 FluidBuffers.hpp
 
 Provide input and output buffering
 
 */
#pragma once

#include <cassert>
#include "FluidTensor.hpp"

namespace fluid{
    
    /*!
     FluidSource
     
     Input buffer, with possibly overlapped reads
     */
    template <typename T>
    class FluidSource//: public FluidTensor<T,2>
    {
        using tensor_type = FluidTensor<T,2>;
        using view_type =   FluidTensorView<T,2>;
        using const_view_type = const FluidTensorView<T,2>;
        
    public:
        FluidSource() = delete;
        FluidSource(FluidSource&) = delete;
        FluidSource& operator=(FluidSource&)=delete;
        
        FluidSource(const size_t size, const size_t channels = 1):
        matrix(channels,size), m_size(size), m_channels(channels)
        {}
        
        tensor_type& data()
        {
            return matrix;
        }
        
        /*
         Push a frame of data into the buffer
         */
        void push(const_view_type x)
        {
            assert(x.rows() == m_channels);
            
            size_t blocksize = x.cols();
            
            assert(blocksize <= buffer_size());
            
            size_t offset = m_counter;
            
            
            size_t size = ((offset + blocksize) > buffer_size()) ?  buffer_size() - offset : blocksize ;
            
            //Copy all channels (rows)
            copy_in(x(slice(0),slice(0,size)), offset, size);
            copy_in(x(slice(0),slice(size,blocksize-size)), 0, blocksize - size);
        }
        
        template<typename S>
        void push(S** in ,size_t nsamps, size_t nchans)
        {
            assert(nchans = m_channels);
            assert(nsamps <= buffer_size());
            size_t blocksize = nsamps;
            
            size_t offset = m_counter;
            
            size_t size = ((offset + blocksize) > buffer_size()) ?  buffer_size() - offset : blocksize ;
            
            copy_in(in, 0, offset, size);
            copy_in(in, size, 0, blocksize-size);
        }
        
        /*!
         Pull a frame of data out of the buffer.
        */
        void pull(view_type out,size_t frame_time)
        {
            size_t blocksize = out.cols();
            size_t offset = m_host_buffer_size - frame_time ;

            if(offset > buffer_size())
            {
                out.fill(0);
                return;
            }
            
            offset += blocksize;
            offset = (offset <= m_counter) ? m_counter - offset : m_counter + buffer_size() - offset;

            size_t size = (offset + blocksize > buffer_size()) ? buffer_size() - offset : blocksize;

            out(slice(0),slice(0,size)) = matrix(slice(0),slice(offset,size));
            out(slice(0),slice(size,blocksize-size)) = matrix(slice(0),slice(0,blocksize-size));
        }
        
        /*
         Set the buffer size of the enclosing host.
         Needed to properly handle latency, causality etc
         */
        void set_host_buffer_size(const size_t size)
        {
            m_host_buffer_size = size;
        }
        
        /*
         Reset the buffer, resizing if the desired
         size and / or host buffer size have changed
         
         This should be called in the DSP setup routine of
         the audio host
         */
        void reset()
        {
            if(matrix.cols() != buffer_size())
                matrix.resize(m_channels,buffer_size());
            matrix.fill(0);
            m_counter = 0;
        }
        
        void set_size(size_t n)
        {
            m_size = n;
        }
        
    private:
        
        /*
         Report the size of the whole buffer
         */
        size_t buffer_size() const
        {
            return m_size + m_host_buffer_size;
        }
        
        /*
          Copy a frame into the buffer and move the write head on
         */
        void copy_in(const_view_type input, const size_t offset, const size_t size)
        {
            if(size)
            {
                matrix(slice(0),slice(offset,size)) = input;
                m_counter = offset + size;
            }
        }
        
        template <typename S>
        void copy_in(S** in, const size_t in_start,const size_t offset, const size_t size)
        {
            if(size)
            {
                for(size_t i = 0; i < m_channels; ++i)
                {
                    
                    auto in_range = matrix(i,slice(offset,size));
                    in[i]->copy_from(in_range, in_start, size);
                }
                m_counter = offset + size;
            }
        }
        
        tensor_type matrix;
        size_t m_counter = 0;
        size_t m_size;
        size_t m_channels;
        size_t m_host_buffer_size;
    };
    
    /*
     FluidSink
     
     An output buffer, with overlap-add
     */
    template <typename T>
    class FluidSink
    {
        using tensor_type = FluidTensor<T,2>;
        using view_type =   FluidTensorView<T,2>;
        using const_view_type = const FluidTensorView<T,2>;
    public:
        FluidSink():FluidSink(65536,1)
        {}
        
        FluidSink(FluidSink&)=delete;
        FluidSink operator=(FluidSink&)=delete;
        
        FluidSink(const size_t size,const size_t channels=1):
        matrix(channels,size), m_size(size), m_channels(channels)
        {}
        
        tensor_type& data()
        {
            return matrix;
        }
        
        /**
         Accumulate data into the buffer, optionally moving
         the write head on by a custom amount.
         
         This *adds* the content of the frame to whatever is
         already there
         **/
        void push(const_view_type x, size_t frame_time)
        {
            assert(x.rows() == m_channels);
            
            size_t blocksize = x.cols();
            
            assert(blocksize <= buffer_size());
            
            size_t offset = frame_time;

            if(offset + blocksize > buffer_size())
            {
                return;
            }
            
            offset += m_counter;
            offset = offset < buffer_size() ? offset : offset - buffer_size();

            size_t size = ((offset + blocksize) > buffer_size()) ? buffer_size() - offset : blocksize;
            
            add_in(x(slice(0), slice(0,size)), offset, size);
            add_in(x(slice(0), slice(size,blocksize-size)), 0, blocksize - size);
        }
        
        /**
         Copy data from the buffer, and zero where it was
         **/
        void pull(view_type out)
        {
            size_t blocksize = out.cols();
            if(blocksize > buffer_size())
            {
                return;
            }
            
            size_t offset = m_counter;
            
            size_t size = offset + blocksize > buffer_size() ? buffer_size() - offset : blocksize;
            
            out_and_zero(out(slice(0), slice(0,size)), offset, size);
            out_and_zero(out(slice(0), slice(size,blocksize-size)), 0, blocksize-size);
        }
        
        template <typename U>
        void pull(U** out, size_t n_samps, size_t n_chans)
        {
            size_t blocksize = n_samps;
            if(blocksize > buffer_size())
            {
                return;
            }
            
            size_t offset = m_counter;
            size_t size = offset + blocksize > buffer_size() ? buffer_size() - offset : blocksize;

            out_and_zero(out, 0, offset, size);
            out_and_zero(out, size, 0,blocksize-size);
            
        }
        
        
        /*!
         Reset the buffer, resizing if the host buffer size
         or user buffer size have changed.
         
         This should be called from an audio host's DSP setup routine
         **/
        void reset()
        {
            if(matrix.cols() != buffer_size())
                matrix.resize(m_channels,buffer_size());
            matrix.fill(0);
            m_counter  = 0;
        }
        
        void set_size(size_t n)
        {
            m_size = n;
        }
        
        void set_host_buffer_size(size_t n)
        {
            m_host_buffer_size = n;
        }
        
    private:

        void add_in(const_view_type in, size_t offset, size_t size)
        {
            if(size)
            {
                matrix(slice(0),slice(offset,size))
                    .apply(in, [](double& x, double y)
                    {
                        x+=y;
                    });
            }
        }
        
        void out_and_zero(view_type out, size_t offset, size_t size)
        {
            if(size)
            {
                view_type buf = matrix(slice(0),slice(offset, size));
                view_type output = buf;
                out = output;
                buf.fill(0);
                m_counter = offset + size;
            }
        }
        
        template <typename U>
        void out_and_zero(U** out, size_t out_offset, size_t offset, size_t size)
        {
            if(size)
            {
                for(size_t i = 0; i < m_channels; ++i)
                {
                    auto out_slice = matrix(i, slice(offset, size));
                    out[i]->copy_to(out_slice,out_offset,size);
                 }
                matrix(slice(0),slice(offset,size)).fill(0);
                m_counter = offset + size;
            }
        }
        
        
        size_t buffer_size() const
        {
            return m_size + m_host_buffer_size;
        }
        
        tensor_type matrix;
        size_t m_size;
        size_t m_channels;
        size_t m_counter = 0;
        size_t m_host_buffer_size = 0;
    };
}
