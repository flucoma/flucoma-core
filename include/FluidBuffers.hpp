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
        void push(FluidTensorView<T,N>& ft);
        FluidTensorView<T,N> pull(size_t size);
        size_t offset;
    };
    
}
