#include "util/audiofile.hpp"
#include "algorithms/STFT.hpp"
#include "data/FluidTensor.hpp"

#include <iostream>
#include <chrono>
#include <ctime>
/**!
 Semantically, it's nicer if we can write functions that return
 output types, rather than using output parameters. Indeed, the CPP core
 guidelines reccomend this, but only in cases where a move operation
 for the type in question is cheap.
 
 For Eigen types, we have the possibilites below, given 'EigenThing' standing in
 as a general Eigen type (Array / Matrix), and Ref<EigenThing> using Eigen::Ref.
 
 __Input parameters of EigenThings should always be refs, to avoid new allocations__
 
 1) EigenThing foo(Ref<const EigenThing>& input) : just return an EigenThing, hope that return value optimisation (RVO) will do its thing and compile to a move operation
 2) Ref<EigenThing> foo(Ref<const EigenThing>& input): return a ref
 3) void foo(Ref<const EigenThing>& input, Ref<EigenThing>& output)
 
 Other factors:
 - These are functions (like FFT.process()) that will be called a lot in an inner processing loop
 – So we want a output buffer allocated at class scope, rather than every call (how does this jive with RVO?)
 – Eventually these are always going to convert back to FluidTensor types, via EigenMaps. FluidTensors are row-major only, Eigen types are column-major by default (and seem to be slower in row-major). Does the cost of going column-major to row-major on the conversion have an impact? What about for the processing?
 
 **/




template <typename T, typename U=T>
class EigenReturnProfiler
{
    using ref = Eigen::Ref<T>;
    using const_ref = const Eigen::Ref<const U>;
public:
    EigenReturnProfiler(size_t size):m_output(size), m_window(size)
    {}
    
    T   f_plain(const_ref input)
    {
        T output = input * m_window;
        return output;
    }
    
    ref f_ref(const_ref input)
    {
        T output = input * m_window;
        return output;
    }
    
    
    T f_class_alloc(const_ref input)
    {
        m_output = input * m_window;
        return m_output;
    }
    
    T& f_class_alloc_native_ref(const_ref input)
    {
        m_output = input * m_window;
        return m_output;
    }
    
    
    ref f_class_alloc_ref(const_ref input)
    {
        m_output = input * m_window;
        return m_output;
    }
    
    //template <typename DerivedIn, typename DerviedOut>
    void f_output_parameter(const_ref input, ref output)
    {
        output = input * m_window;
    }
    
    template <typename Derived>
    ref f_class_alloc_ref_passbase(const Eigen::ArrayBase<Derived>& input)
    {
        m_output = input * m_window;
        return m_output;
    }
    
    
    template <typename DerivedIn, typename DerivedOut>
    void f_output_parameter_passbase(const Eigen::ArrayBase<DerivedIn>& input, Eigen::ArrayBase<DerivedOut>& output)
    {
        output = input * m_window;
    }
    
    T m_output;
    T m_window;
};



int main(int argc, char* argv[])
{
    
    using Eigen::Ref;
    
    constexpr size_t  size = 1024 ;
    constexpr size_t iterations = 10000000;
    using std::chrono::system_clock;
    using std::chrono::duration;
    using eigen_type_col_major = Eigen::ArrayXd;
    using eigen_type_row_major = Eigen::Array<double, 1,Eigen::Dynamic,Eigen::RowMajor>;
    
    
    //    const real_vector test_src_vec(size);
    //    const const_map_type src(test_src_vec.data(),1,size);
    //
    //    real_vector test_dst_vec(size);
    //    map_type dst(test_dst_vec.data(),1,size);
    //
    using std::chrono::system_clock;
    using std::chrono::duration;
    
    using real_vector = fluid::FluidTensor<double, 1>;
    
    using map_type = Eigen::Map<Eigen::Array<double, 1,Eigen::Dynamic,Eigen::RowMajor>>;
    using const_map_type = Eigen::Map<const Eigen::Array<double, 1,Eigen::Dynamic,Eigen::RowMajor>>;
    
    const real_vector test_src_vec(size);
    const const_map_type src(test_src_vec.data(),1,size);
    
    real_vector test_dst_vec(size);
    map_type dst(test_dst_vec.data(),1,size);
    
    
    EigenReturnProfiler<eigen_type_col_major> test_colwise(size);
    
    //Plain
    auto start = system_clock::now();
    for(int i = 0; i < iterations; ++i)
        dst = test_colwise.f_plain(src);
    auto end = system_clock::now();
    duration<double> dur = end - start;
    std::cout << "Colwise Plain Version: " << dur.count() << '\n';
    
    //plain Rerturn Ref
    start = system_clock::now();
    for(int i = 0; i < iterations; ++i)
        dst = test_colwise.f_ref(src);
    end = system_clock::now();
    dur = end - start;
    std::cout << "Colwise Plain Return Ref: " << dur.count() << '\n';
    
    // prealloc plain return
    start = system_clock::now();
    for(int i = 0; i < iterations; ++i)
        dst = test_colwise.f_class_alloc(src);
    end = system_clock::now();
    dur = end - start;
    std::cout << "Colwise Preallocation Plain Return: " << dur.count() << '\n';
    
    // prealloc ref return
    start = system_clock::now();
    for(int i = 0; i < iterations; ++i)
        dst = test_colwise.f_class_alloc_ref(src);
    end = system_clock::now();
    dur = end - start;
    std::cout << "Colwise Preallocation Eigen Ref Return: " << dur.count() << '\n';

    
    // prealloc native ref return
    start = system_clock::now();
    for(int i = 0; i < iterations; ++i)
        dst = test_colwise.f_class_alloc_native_ref(src);
    end = system_clock::now();
    dur = end - start;
    std::cout << "Colwise Preallocation Native Ref Return: " << dur.count() << '\n';

    
    // templated input
    start = system_clock::now();
    for(int i = 0; i < iterations; ++i)
        dst = test_colwise.f_class_alloc_ref_passbase(src);
    end = system_clock::now();
    dur = end - start;
    std::cout << "Colwise Preallocation Ref Return Arraybase Template Fn: " << dur.count() << '\n';


    // outputparam
    start = system_clock::now();
    for(int i = 0; i < iterations; ++i)
        test_colwise.f_output_parameter(src,dst);
    end = system_clock::now();
    dur = end - start;
    std::cout << "Colwise Output Param: " << dur.count() << '\n';

    // outputparam template fn
    start = system_clock::now();
    for(int i = 0; i < iterations; ++i)
        test_colwise.f_output_parameter_passbase(src,dst);
    end = system_clock::now();
    dur = end - start;
    std::cout << "Colwise Output Param template func: " << dur.count() << '\n';

    
    
    EigenReturnProfiler<eigen_type_row_major> test_rowwise(size);
    
    //Plain
    start = system_clock::now();
    for(int i = 0; i < iterations; ++i)
        dst = test_rowwise.f_plain(src);
    end = system_clock::now();
    dur = end - start;
    std::cout << "Rowwise Plain Version: " << dur.count() << '\n';
    
    //plain Rerturn Ref
    start = system_clock::now();
    for(int i = 0; i < iterations; ++i)
        dst = test_rowwise.f_ref(src);
    end = system_clock::now();
    dur = end - start;
    std::cout << "Rowwise Plain Return Ref: " << dur.count() << '\n';
    
    // prealloc plain return
    start = system_clock::now();
    for(int i = 0; i < iterations; ++i)
        dst = test_rowwise.f_class_alloc(src);
    end = system_clock::now();
    dur = end - start;
    std::cout << "Rowwise Preallocation Plain Return: " << dur.count() << '\n';
    
    // prealloc ref return
    start = system_clock::now();
    for(int i = 0; i < iterations; ++i)
        dst = test_rowwise.f_class_alloc_ref(src);
    end = system_clock::now();
    dur = end - start;
    std::cout << "Rowwise Preallocation Ref Return: " << dur.count() << '\n';
    
    // prealloc native ref return
    start = system_clock::now();
    for(int i = 0; i < iterations; ++i)
        dst = test_rowwise.f_class_alloc_native_ref(src);
    end = system_clock::now();
    dur = end - start;
    std::cout << "Rowwise Preallocation Native Ref Return: " << dur.count() << '\n';
    
    // prealloc ref return
    start = system_clock::now();
    for(int i = 0; i < iterations; ++i)
        dst = test_rowwise.f_class_alloc_ref_passbase(src);
    end = system_clock::now();
    dur = end - start;
    std::cout << "Rowwise Preallocation Ref Return ArrayBase Template Fn: " << dur.count() << '\n';
    
    // outputparam
    start = system_clock::now();
    for(int i = 0; i < iterations; ++i)
        test_rowwise.f_output_parameter(src,dst);
    end = system_clock::now();
    dur = end - start;
    std::cout << "Rowwise Preallocation Ref Return: " << dur.count() << '\n';
    
    
    // outputparam
    start = system_clock::now();
    for(int i = 0; i < iterations; ++i)
        test_rowwise.f_output_parameter(src,dst);
    end = system_clock::now();
    dur = end - start;
    std::cout << "Rowwise Output Param: " << dur.count() << '\n';
    
    // outputparam template fn
    start = system_clock::now();
    for(int i = 0; i < iterations; ++i)
        test_rowwise.f_output_parameter_passbase(src,dst);
    end = system_clock::now();
    dur = end - start;
    std::cout << "Rowwise Output Param template func: " << dur.count() << '\n';

    
    return 0;
    
}




