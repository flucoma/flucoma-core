#include <data/FluidTensor.hpp> 

//None of these should compile
int main() 
{
    #ifdef FAIL_CONSTRUCT_CANT_CONVERT 
        struct No{}; 
        fluid::FluidTensor<No,1> x(); 
        fluid::FluidTensor<int,1> y(x); 
    #endif
    
    #ifdef CONFIRM_CONSTRUCT_CONVERT         
        fluid::FluidTensor<long,1> x(); 
        fluid::FluidTensor<int,1> y(x); 
    #endif

    #ifdef CONFIRM_CONSTRUCT_ENOUGH_DIMS
        fluid::FluidTensor<int,2>(3,2); 
    #endif 

    #ifdef FAIL_CONSTRUCT_NOT_ENOUGH_DIMS
        fluid::FluidTensor<int,2>(3); 
    #endif 

    #ifdef FAIL_CONSTRUCT_TOO_MANY_DIM
        fluid::FluidTensor<int,2>(3); 
    #endif 

    #ifdef CONFIRM_ASSIGN_CORRECT_DIM 
        fluid::FluidTensor<int,2> x(3); 
        fluid::FluidTensor<int,1> y(3); 
        y = x; 
    #endif 

    #ifdef FAIL_ASSIGN_TOO_MANY_DIM 
        fluid::FluidTensor<int,2> x(3,3); 
        fluid::FluidTensor<int,1> y(3); 
        y = x; 
    #endif 

    #ifdef CONFIRM_ASSIGN_CONVERT 
        fluid::FluidTensor<long,1> x(); 
        fluid::FluidTensor<int,1> y(); 
        y = x; 
    #endif

    #ifdef FAIL_ASSIGN_CANT_CONVERT 
        struct No{}; 
        fluid::FluidTensor<No,1> x(); 
        fluid::FluidTensor<int,1> y(); 
        y = x; 
    #endif

    #ifdef CONFIRM_ACCESS_INDEX_CORRECT_DIMS
        fluid::FluidTensor<int,1> x(); 
        x(1); 
    #endif 

    #ifdef FAIL_ACCESS_INDEX_WRONG_DIMS
        fluid::FluidTensor<int,1> x(); 
        x(1,2,3); 
    #endif 

    #ifdef CONFIRM_ACCESS_SLICE_CORRECT_DIMS
        fluid::FluidTensor<int,2> x(); 
        x(fluid::Slice(0),3); 
    #endif 

    #ifdef FAIL_ACCESS_SLICE_WRONG_DIMS
        fluid::FluidTensor<int,1> x(); 
        x(fluid::Slice(1),2,3); 
    #endif 

    #ifdef CONFIRM_RESIZE_CORRECT_DIMS
        fluid::FluidTensor<int,3> x(); 
        x.resize(1,2,3); 
    #endif 

    #ifdef FAIL_RESIZE_WRONG_DIMS
        fluid::FluidTensor<int,1> x(); 
        x.resize(1,2,3); 
    #endif 
}