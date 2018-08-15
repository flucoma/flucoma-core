/*!
 FluidBuffers.hpp
 
 Provide input and output buffering
 
 */
#pragma once

#include "FluidTensor.hpp"

namespace fluid{
    
    template <typename T, size_t N>
    class FluidSource: public FluidTensor<T,N>
    {
    public:
        FluidSource() = delete;
        FluidSource(FluidSource&) = delete;
        FluidSource& operator=(FluidSource&)=delete;
        
        template<typename... Dims,
        typename = enable_if_t<is_index_sequence<Dims...>()>>
        FluidSource(Dims ...dims) {}
        
        void push(FluidTensorView<T,N>& ft)
        {
            
        }
        FluidTensorView<T,N> pull(size_t size);
        
        size_t available()
        {
            return 0;
        }
    
    private: 
        size_t m_read_head;
        size_t m_write_head;
    };
    
    
    
    
}
