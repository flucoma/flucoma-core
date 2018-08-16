/*!
 FluidBuffers.hpp
 
 Provide input and output buffering
 
 */
#pragma once

#include <cassert>
#include "FluidTensor.hpp"

namespace fluid{
    template <typename T>
    class FluidSource//: public FluidTensor<T,2>
    {
        using tensor_type = FluidTensor<T,2>;
        using view_type =   FluidTensorView<T,2>&;
        using const_view_type = const FluidTensorView<T,2>&;
        
    public:
        FluidSource() = delete;
        FluidSource(FluidSource&) = delete;
        FluidSource& operator=(FluidSource&)=delete;
        
        FluidSource(const size_t size, const size_t channels = 1):
        matrix(channels, size), m_size(size), m_channels(channels)
        {}
        

        
        void push(const_view_type x)
        {
            assert(x.rows() == m_channels);
            
            size_t blocksize = x.cols();
            
            assert(blocksize <= m_size);
            
            size_t size = ((m_write_head + blocksize) > m_size) ? m_size - m_write_head : blocksize;
            
            copy_in(x(0,slice(0,size)), m_write_head, size);
            copy_in(x(0,slice(size,blocksize-size)), 0, blocksize - size);
        }
        
        tensor_type pull(const size_t blocksize, const size_t hop = 0)
        {
            
            //TODO think properly about this. We're always going to return requested size
            //for now, padded with 0s
            //assert(blocksize <= available());
            
            tensor_type out(1,blocksize);
            
            size_t availsize = blocksize <= available()? blocksize:available();
            
            
            size_t size = m_read_head + availsize > m_size ? m_size - m_read_head : availsize;
            
            out(0,slice(0,size)) = matrix(0,slice(m_read_head,size));
            out(0,slice(size,availsize-size)) = matrix(0,slice(0,availsize-size));
            
            size_t advance = hop ? hop : availsize;
            
            m_read_head = m_read_head + advance > m_size ?
                m_size - m_read_head + advance: m_read_head + advance;
            
            return out;
        }
        
        void reset()
        {
            matrix.fill(0);
            m_write_head = 0;
            m_read_head  = 0;
        }
        
        size_t available()
        {
            return m_write_head >= m_read_head ?
                m_write_head - m_read_head : m_size - m_read_head + m_write_head;
        }
    
        void resize(size_t n)
        {
            matrix.resize(1,n);
            m_size = n; 
            reset(); 
        }
        
    private:
        
        void copy_in(const_view_type input, const size_t offset, const size_t size)
        {
            if(size)
            {
                matrix(0, slice(offset,size)) = input;
                m_write_head = offset + size;
            }
        }
        
        void copy_out(view_type out, const size_t offset, const size_t size)
        {
            if(size)
            {
                out(0,slice(0,size)) = matrix(0,slice(offset,size));
                m_read_head = offset + size; 
            }
        }
        
        tensor_type matrix;
        size_t m_read_head = 0;
        size_t m_write_head = 0;
//        size_t m_delay_time = 0;
        size_t m_size;
        size_t m_channels;
    };
    
    template <typename T>
    class FluidSink
    {
        using tensor_type = FluidTensor<T,2>;
        using view_type =   FluidTensorView<T,2>;
        using const_view_type = const FluidTensorView<T,2>;
    public:
        FluidSink():FluidSink(1)
        {}
        
        FluidSink(FluidSink&)=delete;
        FluidSink operator=(FluidSink&)=delete;
        
        FluidSink(size_t channels, size_t size=65536):
        matrix(channels, size), m_size(size), m_channels(channels)
        {}
        
        
        void push(const_view_type x, size_t hop = 0)
        {
            assert(x.rows() == m_channels);
            
            size_t blocksize = x.cols();
            
            assert(blocksize <= m_size);
            
            size_t size = ((m_write_head + blocksize) > m_size) ? m_size - m_write_head : blocksize;
            
            add_in(x(0,slice(0,size)), m_write_head, size);
            add_in(x(0,slice(size,blocksize-size)), 0, blocksize - size);
            
            size_t advance = hop ? hop : blocksize;
            
            m_write_head = m_write_head + advance > m_size ?
                 m_size - m_write_head + advance: m_write_head + advance;
            
        }
        
        tensor_type pull(size_t blocksize)
        {
            //assert(blocksize <= available());
            
            tensor_type out(1,blocksize);
            
            size_t availsize = blocksize <= available()? blocksize:available();
            
            size_t size = m_read_head + availsize > m_size ? m_size - m_read_head : availsize;
            
            out_and_zero(out(0,slice(0,size)), m_read_head, size);
            out_and_zero(out(0,slice(size,availsize-size)), 0, availsize-size);
            
            return out;
        }
        
        void reset()
        {
            matrix.fill(0);
            m_write_head = 0;
            m_read_head  = 0;
        }
        
        size_t available()
        {
            return m_write_head >= m_read_head ?
            m_write_head - m_read_head : m_size - m_read_head + m_write_head;
        }
        
        void resize(size_t n)
        {
            matrix.resize(1,n);
            m_size = n;
            reset();
        }
        
        
    private:

        void add_in(const_view_type in, size_t offset, size_t size)
        {
            if(size)
            {
                matrix(0,slice(offset,size))
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
                view_type output = matrix(0,slice(offset, size));
                out = output;
                output.fill(0);
                m_read_head = offset + size;
            }
        }
        
        tensor_type matrix;
        size_t m_size;
        size_t m_channels;
        size_t m_read_head = 0;
        size_t m_write_head = 0;
    };
}
