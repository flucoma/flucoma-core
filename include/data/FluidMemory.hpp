#pragma once

#include <Eigen/Core> 
#include <memory/allocator_storage.hpp>
#include <memory/container.hpp>
#include <memory/heap_allocator.hpp>

namespace fluid {

    
    // using MappedMatrix = Eigen::Map<MatrixXXd>; 
    // using MappedArray = Eigen::Map<MatrixXXd>; 

    
    using Allocator  = foonathan::memory::any_allocator_reference;    

    template<typename T>
    using RTVector = foonathan::memory::vector<T,Allocator>;

    template<typename T>
    using RTDeque = foonathan::memory::deque<T,Allocator>;

    Allocator& FluidDefaultAllocator() 
    {
        static Allocator def = foonathan::memory::make_allocator_reference(foonathan::memory::heap_allocator()); 
        return def; 
    }

}
