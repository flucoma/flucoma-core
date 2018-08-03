
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <sstream>
#include <iostream>
#include <iomanip>

#include "HISSTools_FFT.h"

// Output

void tabbedOut(const std::string& name, const std::string& text, int tab = 25)
{
    std::cout << std::setw(tab) << std::setfill(' ');
    std::cout.setf(std::ios::left);
    std::cout.unsetf(std::ios::right);
    std::cout << name;
    std::cout.unsetf(std::ios::left);
    std::cout << text << "\n";
}

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 4, bool fixed = true)
{
    std::ostringstream out;
    if (fixed)
        out << std::setprecision(n) << std::fixed << a_value;
    else
        out << std::setprecision(n) << a_value;
    
    return out.str();
}

// Timing

class Timer
{
    
public:
    
    Timer() : mStart(0), mStore1(0), mStore2(0) {}
    
    void start()
    {
        mStart = mach_absolute_time();
    };
    
    void stop(const std::string& msg)
    {
        uint64_t end = mach_absolute_time();
        
        mach_timebase_info_data_t info;
        mach_timebase_info(&info);
        
        uint64_t elapsed = ((end - mStart) * info.numer) / info.denom;
        tabbedOut(msg + " Elapsed ", to_string_with_precision(elapsed / 1000000.0, 2), 35);
        
        mStore2 = mStore1;
        mStore1 = elapsed;
    };
    
    void relative(const std::string& msg)
    {
        tabbedOut(msg + " Comparison ", to_string_with_precision(((double) mStore1 / (double) mStore2), 2), 35);
    }
    
private:
    
    uint64_t        mStart;
    uint64_t        mStore1;
    uint64_t        mStore2;
};

template<class SETUP, class SPLIT, class T>void crash_test(int min_log2, int max_log2)
{
    SETUP setup;
    SPLIT split;
    
    split.realp = (T *) malloc(sizeof(T) * 1 << max_log2);
    split.imagp = (T *) malloc(sizeof(T) * 1 << max_log2);
    
    hisstools_create_setup(&setup, max_log2);

    Timer timer;
    timer.start();

    for (int i = min_log2; i < max_log2; i++)
        hisstools_fft(setup, &split, i);
    
    for (int i = min_log2; i < max_log2; i++)
        hisstools_ifft(setup, &split, i);
    
    for (int i = min_log2; i < max_log2; i++)
        hisstools_rfft(setup, &split, i);
    
    for (int i = min_log2; i < max_log2; i++)
        hisstools_rifft(setup, &split, i);
    
    timer.stop("FFT Multiple Tests");

    free(split.realp);
    free(split.imagp);
    hisstools_destroy_setup(setup);
}

template<class SETUP, class SPLIT, class T> void single_test(int size, void (*Fn)(SETUP, SPLIT *, uintptr_t))
{
    SETUP setup;
    SPLIT split;
    
    split.realp = (T *) malloc(sizeof(T) * 1 << size);
    split.imagp = (T *) malloc(sizeof(T) * 1 << size);
    
    hisstools_create_setup(&setup, size);
    
    Timer timer;
    timer.start();
    for (int i = 0; i < 10000; i++)
        Fn(setup, &split, size);
    timer.stop("FFT Single Tests");

    free(split.realp);
    free(split.imagp);
    hisstools_destroy_setup(setup);
}

template<class SETUP, class SPLIT, class T>void matched_size_test(int min_log2, int max_log2)
{
    for (int i = min_log2; i < max_log2; i++)
        single_test<SETUP, SPLIT, T>(i, &hisstools_fft);
    
    for (int i = min_log2; i < max_log2; i++)
        single_test<SETUP, SPLIT, T>(i, &hisstools_ifft);
    
    for (int i = min_log2; i < max_log2; i++)
        single_test<SETUP, SPLIT, T>(i, &hisstools_rfft);
    
    for (int i = min_log2; i < max_log2; i++)
        single_test<SETUP, SPLIT, T>(i, &hisstools_ifft);
}

template<class SPLIT, class T, class U>void zip_test(int min_log2, int max_log2)
{
    SPLIT split;
    
    U *ptr = (U *) malloc(sizeof(U) * 1 << max_log2);
    split.realp = (T *) malloc(sizeof(T) * 1 << (max_log2 - 1));
    split.imagp = (T *) malloc(sizeof(T) * 1 << (max_log2 - 1));
    
    Timer timer;
    timer.start();
    
    for (int i = min_log2; i < max_log2; i++)
        hisstools_zip(&split, ptr, i);
    
    for (int i = min_log2; i < max_log2; i++)
        hisstools_unzip(ptr, &split, i);
    
    for (int i = min_log2; i < max_log2; i++)
        hisstools_unzip_zero(ptr, &split, 1 << i, i);
    
    timer.stop("Zip Tests");

    free(split.realp);
    free(split.imagp);
}


int main(int argc, const char * argv[])
{
    Timer timer;
    
    timer.start();
    
    crash_test<FFT_SETUP_D, FFT_SPLIT_COMPLEX_D, double>(0, 22);
    crash_test<FFT_SETUP_F, FFT_SPLIT_COMPLEX_F, float>(0, 22);
    
    matched_size_test<FFT_SETUP_D, FFT_SPLIT_COMPLEX_D, double>(6, 14);
    matched_size_test<FFT_SETUP_F, FFT_SPLIT_COMPLEX_F, float>(6, 14);
    
    zip_test<FFT_SPLIT_COMPLEX_D, double, double>(1, 22);
    zip_test<FFT_SPLIT_COMPLEX_F, float, float>(1, 22);
    
    timer.stop("FFT Crash Tests Total");
    
    std::cout << "Finished Running\n";
    return 0;
}
