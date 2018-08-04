
#ifndef __APPLE__
#include <intrin.h>
#endif

#include <cmath>
#include <algorithm>
#include <functional>
#include <emmintrin.h>
#include <immintrin.h>

// Setup Structures

template <class T> struct Setup
{
    unsigned long max_fft_log2;
    Split<T> tables[28];
};

struct DoubleSetup : public Setup<double> {};
struct FloatSetup : public Setup<float> {};

#define SIMD_COMPILER_SUPPORT_SCALAR 0
#define SIMD_COMPILER_SUPPORT_SSE128 1
#define SIMD_COMPILER_SUPPORT_AVX256 2
#define SIMD_COMPILER_SUPPORT_AVX512 3

#if defined(__AVX512F__)
#define SIMD_COMPILER_SUPPORT_LEVEL SIMD_COMPILER_SUPPORT_AVX512
#elif defined(__AVX__)
#define SIMD_COMPILER_SUPPORT_LEVEL SIMD_COMPILER_SUPPORT_AVX256
#elif defined(__SSE__)
#define SIMD_COMPILER_SUPPORT_LEVEL SIMD_COMPILER_SUPPORT_SSE128
#else
#define SIMD_COMPILER_SUPPORT_LEVEL SIMD_COMPILER_SUPPORT_SCALAR
#endif

namespace hisstools_fft_impl{

    // Aligned Allocation and Platform CPU Detection
    
#ifdef __APPLE__
    
#include <cpuid.h>

    template <class T> T *allocate_aligned(size_t size)
    {
        return static_cast<T *>(malloc(size * sizeof(T)));
    }
    
    template <class T> void deallocate_aligned(T *ptr)
    {
        free(ptr);
    }
    
    void cpuid(int32_t out[4], int32_t x)
    {
        __cpuid_count(x, 0, out[0], out[1], out[2], out[3]);
    }
    
	uint64_t xgetbv(unsigned int index)
	{
		uint32_t eax, edx;
		__asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
		return ((uint64_t)edx << 32) | eax;
	}

#else
#include <malloc.h>

    template <class T> T *allocate_aligned(size_t size)
    {
        return static_cast<T *>(_aligned_malloc(size * sizeof(T), 16));
    }
    
    template <class T> void deallocate_aligned(T *ptr)
    {
        _aligned_free(ptr);
    }

    void cpuid(int32_t out[4], int32_t x)
    {
        __cpuid(out, x);
    }

	uint64_t xgetbv(unsigned int x) 
	{
		return _xgetbv(x);
	}
    
#endif

    // Offset for Table
    
    static const uintptr_t trig_table_offset = 3;
    
    // CPU Detection
    
    enum SIMDType { kNone, kSSE, kAVX256, kAVX512 };
    
    extern SIMDType SIMD_Support;

    SIMDType detect_SIMD()
    {
        int cpu_info[4] = {-1, 0, 0, 0};
        
        cpuid(cpu_info, 0);
        
        if (cpu_info[0] <= 0)
            return kNone;
        
        cpuid(cpu_info, 1);
        
        if ((cpu_info[3] >> 26) & 0x1)
        {
            bool os_uses_xsave_xrstore = (cpu_info[2] & (1 << 27)) != 0;
            bool cpu_AVX_suport = (cpu_info[2] & (1 << 28)) != 0;
            
            if (os_uses_xsave_xrstore && cpu_AVX_suport)
            {
                uint64_t xcr_feature_mask = xgetbv(0);
                
                if ((xcr_feature_mask & 0x6) == 0x6)
                {
                    if ((xcr_feature_mask & 0xe6) == 0xe6)
                        return kAVX512;
                    else
                        return kAVX256;
                }
            }
            
            return kSSE;
        }
        
        return kNone;
    }
    
    // Data Type Definitions
    
    // ******************** Basic Data Type Defintions ******************** //
    
    template <class T> struct Scalar
    {
        static const int size = 1;
        typedef T scalar_type;
        typedef Split<scalar_type> split_type;
        typedef Setup<scalar_type> setup_type;
        
        Scalar() {}
        Scalar(T a) : mVal(a) {}
        friend Scalar operator + (const Scalar& a, const Scalar& b) { return Scalar(a.mVal + b.mVal); }
        friend Scalar operator - (const Scalar& a, const Scalar& b) { return Scalar(a.mVal - b.mVal); }
        friend Scalar operator * (const Scalar& a, const Scalar& b) { return Scalar(a.mVal * b.mVal); }
        
        T mVal;
    };
    
    template <class T, class U, int vec_size> struct SIMDVector
    {
        static const int size = vec_size;
        typedef T scalar_type;
        typedef Split<scalar_type> split_type;
        typedef Setup<scalar_type> setup_type;
        
        SIMDVector() {}
        SIMDVector(U a) : mVal(a) {}
        
        U mVal;
    };
    
#if (SIMD_COMPILER_SUPPORT_LEVEL >= SIMD_COMPILER_SUPPORT_SSE)
    
    struct SSEDouble : public SIMDVector<double, __m128d, 2>
    {
        SSEDouble() {}
        SSEDouble(__m128d a) : SIMDVector(a) {}
        friend SSEDouble operator + (const SSEDouble &a, const SSEDouble& b) { return _mm_add_pd(a.mVal, b.mVal); }
        friend SSEDouble operator - (const SSEDouble &a, const SSEDouble& b) { return _mm_sub_pd(a.mVal, b.mVal); }
        friend SSEDouble operator * (const SSEDouble &a, const SSEDouble& b) { return _mm_mul_pd(a.mVal, b.mVal); }
        
        template <int y, int x> static SSEDouble shuffle(const SSEDouble& a, const SSEDouble& b)
        {
            return _mm_shuffle_pd(a.mVal, b.mVal, (y<<1)|x);
        }
    };
    
    struct SSEFloat : public SIMDVector<float, __m128, 4>
    {
        SSEFloat() {}
        SSEFloat(__m128 a) : SIMDVector(a) {}
        friend SSEFloat operator + (const SSEFloat& a, const SSEFloat& b) { return _mm_add_ps(a.mVal, b.mVal); }
        friend SSEFloat operator - (const SSEFloat& a, const SSEFloat& b) { return _mm_sub_ps(a.mVal, b.mVal); }
        friend SSEFloat operator * (const SSEFloat& a, const SSEFloat& b) { return _mm_mul_ps(a.mVal, b.mVal); }
        
        template <int z, int y, int x, int w> static SSEFloat shuffle(const SSEFloat& a, const SSEFloat& b)
        {
            return _mm_shuffle_ps(a.mVal, b.mVal, ((z<<6)|(y<<4)|(x<<2)|w));
        }
    };
    
#endif
    
#if (SIMD_COMPILER_SUPPORT_LEVEL >= SIMD_COMPILER_SUPPORT_AVX256)

    struct AVX256Double : public SIMDVector<double, __m256d, 4>
    {
        AVX256Double() {}
        AVX256Double(__m256d a) : SIMDVector(a) {}
        friend AVX256Double operator + (const AVX256Double &a, const AVX256Double &b) { return _mm256_add_pd(a.mVal, b.mVal); }
        friend AVX256Double operator - (const AVX256Double &a, const AVX256Double &b) { return _mm256_sub_pd(a.mVal, b.mVal); }
        friend AVX256Double operator * (const AVX256Double &a, const AVX256Double &b) { return _mm256_mul_pd(a.mVal, b.mVal); }
    };
    
    struct AVX256Float : public SIMDVector<float, __m256, 8>
    {
        AVX256Float() {}
        AVX256Float(__m256 a) : SIMDVector(a) {}
        friend AVX256Float operator + (const AVX256Float &a, const AVX256Float &b) { return _mm256_add_ps(a.mVal, b.mVal); }
        friend AVX256Float operator - (const AVX256Float &a, const AVX256Float &b) { return _mm256_sub_ps(a.mVal, b.mVal); }
        friend AVX256Float operator * (const AVX256Float &a, const AVX256Float &b) { return _mm256_mul_ps(a.mVal, b.mVal); }
    };
    
#endif
    
#if (SIMD_COMPILER_SUPPORT_LEVEL >= SIMD_COMPILER_SUPPORT_AVX512)

    struct AVX512Double : public SIMDVector<double, __m512d, 8>
    {
        AVX512Double() {}
        AVX512Double(__m512d a) : SIMDVector(a) {}
        friend AVX512Double operator + (const AVX512Double &a, const AVX512Double &b) { return _mm512_add_pd(a.mVal, b.mVal); }
        friend AVX512Double operator - (const AVX512Double &a, const AVX512Double &b) { return _mm512_sub_pd(a.mVal, b.mVal); }
        friend AVX512Double operator * (const AVX512Double &a, const AVX512Double &b) { return _mm512_mul_pd(a.mVal, b.mVal); }
    };
    
    struct AVX512Float : public SIMDVector<float, __m512, 16>
    {
        AVX512Float() {}
        AVX512Float(__m512 a) : SIMDVector(a) {}
        friend AVX512Float operator + (const AVX512Float &a, const AVX512Float &b) { return _mm512_add_ps(a.mVal, b.mVal); }
        friend AVX512Float operator - (const AVX512Float &a, const AVX512Float &b) { return _mm512_sub_ps(a.mVal, b.mVal); }
        friend AVX512Float operator * (const AVX512Float &a, const AVX512Float &b) { return _mm512_mul_ps(a.mVal, b.mVal); }
    };
    
#endif
    
    // ******************** A Vector of Given Size (Made of Vectors / Scalars) ******************** //
    
    template <int final_size, class T> struct SizedVector
    {
        static const int size = final_size;
        typedef typename T::scalar_type scalar_type;
        typedef Split<scalar_type> split_type;
        typedef Setup<scalar_type> setup_type;
        static const int array_size = final_size / T::size;
        
        SizedVector() {}
        SizedVector(const SizedVector *ptr) { *this = *ptr; }
        SizedVector(const typename T::scalar_type *array) { *this = *reinterpret_cast<const SizedVector *>(array); }
        
        // This template allows a static loop
        
        template <int First, int Last>
        struct static_for
        {
            template <typename Fn>
            void operator()(SizedVector &result, const SizedVector &a, const SizedVector &b, Fn const& fn) const
            {
                if (First < Last)
                {
                    result.mData[First] = fn(a.mData[First], b.mData[First]);
                    static_for<First + 1, Last>()(result, a, b, fn);
                }
            }
        };
        
        // This specialisation avoids infinite recursion
        
        template <int N>
        struct static_for<N, N>
        {
            template <typename Fn>
            void operator()(SizedVector &result, const SizedVector &a, const SizedVector &b, Fn const& fn) const {}
        };
        
        template <typename Op> friend SizedVector operate(const SizedVector& a, const SizedVector& b, Op op)
        {
            SizedVector result;
            
            static_for<0, array_size>()(result, a, b, op);
            
            return result;
        }
        
        friend SizedVector operator + (const SizedVector& a, const SizedVector& b)
        {
            return operate(a, b, std::plus<T>());
        }
        
        friend SizedVector operator - (const SizedVector& a, const SizedVector& b)
        {
            return operate(a, b, std::minus<T>());
        }
        
        friend SizedVector operator * (const SizedVector& a, const SizedVector& b)
        {
            return operate(a, b, std::multiplies<T>());
        }
        
        T mData[array_size];
    };
    
    // ******************** Setup Creation and Destruction ******************** //
    
    // Creation
    
    template <class T> Setup<T> *create_setup(uintptr_t max_fft_log2)
    {
        // Check for SIMD Support here (this must be called anyway before doing an FFT)
        
        if (SIMD_Support == kNone)
            SIMD_Support = detect_SIMD();
        
        Setup<T> *setup = allocate_aligned<Setup<T> >(1);
        
        // Set Max FFT Size
        
        setup->max_fft_log2 = max_fft_log2;
        
        // Create Tables
        
        for (uintptr_t i = trig_table_offset; i <= max_fft_log2; i++)
        {
            uintptr_t length = (uintptr_t) 1 << (i - 1);
            
            setup->tables[i - trig_table_offset].realp = allocate_aligned<T>(2 * length);
            setup->tables[i - trig_table_offset].imagp = setup->tables[i - trig_table_offset].realp + length;
            
            // Fill the Table
            
            T *table_real = setup->tables[i - trig_table_offset].realp;
            T *table_imag = setup->tables[i - trig_table_offset].imagp;
            
            for (uintptr_t j = 0; j < length; j++)
            {
                static const double pi = 3.14159265358979323846264338327950288;
                double angle = -(static_cast<double>(j)) * pi / static_cast<double>(length);
                
                *table_real++ = static_cast<T>(cos(angle));
                *table_imag++ = static_cast<T>(sin(angle));
            }
        }
        
        return setup;
    }
    
    // Destruction
    
    template <class T> void destroy_setup(Setup<T> *setup)
    {
        if (setup)
        {
            for (uintptr_t i = trig_table_offset; i <= setup->max_fft_log2; i++)
                deallocate_aligned(setup->tables[i - trig_table_offset].realp);
            
            deallocate_aligned(setup);
        }
    }
    
    // ******************** Shuffles for Pass 1 and 2 ******************** //
    
    // Template for an SIMD Vectors With 4 Elements
    
    template <class T>
    void shuffle4(const SizedVector<4, T> &A,
                  const SizedVector<4, T> &B,
                  const SizedVector<4, T> &C,
                  const SizedVector<4, T> &D,
                  SizedVector<4, T> *ptr1,
                  SizedVector<4, T> *ptr2,
                  SizedVector<4, T> *ptr3,
                  SizedVector<4, T> *ptr4)
    {}
    
    // Template for Scalars
    
    template<class T>
    void shuffle4(const SizedVector<4, Scalar<T> > &A,
                  const SizedVector<4, Scalar<T> > &B,
                  const SizedVector<4, Scalar<T> > &C,
                  const SizedVector<4, Scalar<T> > &D,
                  SizedVector<4, Scalar<T> > *ptr1,
                  SizedVector<4, Scalar<T> > *ptr2,
                  SizedVector<4, Scalar<T> > *ptr3,
                  SizedVector<4, Scalar<T> > *ptr4)
    {
        ptr1->mData[0] = A.mData[0];
        ptr1->mData[1] = C.mData[0];
        ptr1->mData[2] = B.mData[0];
        ptr1->mData[3] = D.mData[0];
        ptr2->mData[0] = A.mData[2];
        ptr2->mData[1] = C.mData[2];
        ptr2->mData[2] = B.mData[2];
        ptr2->mData[3] = D.mData[2];
        ptr3->mData[0] = A.mData[1];
        ptr3->mData[1] = C.mData[1];
        ptr3->mData[2] = B.mData[1];
        ptr3->mData[3] = D.mData[1];
        ptr4->mData[0] = A.mData[3];
        ptr4->mData[1] = C.mData[3];
        ptr4->mData[2] = B.mData[3];
        ptr4->mData[3] = D.mData[3];
    }

#if (SIMD_COMPILER_SUPPORT_LEVEL >= SIMD_COMPILER_SUPPORT_SSE)

    // Template Specialisation for an SSE Float Packed (1 SIMD Element)
    
    template<>
    void shuffle4(const SizedVector<4, SSEFloat> &A,
                  const SizedVector<4, SSEFloat> &B,
                  const SizedVector<4, SSEFloat> &C,
                  const SizedVector<4, SSEFloat> &D,
                  SizedVector<4, SSEFloat> *ptr1,
                  SizedVector<4, SSEFloat> *ptr2,
                  SizedVector<4, SSEFloat> *ptr3,
                  SizedVector<4, SSEFloat> *ptr4)
    {
        const SSEFloat v1 = SSEFloat::shuffle<1, 0, 1, 0>(A.mData[0], C.mData[0]);
        const SSEFloat v2 = SSEFloat::shuffle<3, 2, 3, 2>(A.mData[0], C.mData[0]);
        const SSEFloat v3 = SSEFloat::shuffle<1, 0, 1, 0>(B.mData[0], D.mData[0]);
        const SSEFloat v4 = SSEFloat::shuffle<3, 2, 3, 2>(B.mData[0], D.mData[0]);
        
        ptr1->mData[0] = SSEFloat::shuffle<2, 0, 2, 0>(v1, v3);
        ptr2->mData[0] = SSEFloat::shuffle<2, 0, 2, 0>(v2, v4);
        ptr3->mData[0] = SSEFloat::shuffle<3, 1, 3, 1>(v1, v3);
        ptr4->mData[0] = SSEFloat::shuffle<3, 1, 3, 1>(v2, v4);
    }
    
    // Template Specialisation for an SSE Double Packed (2 SIMD Elements)
    
    template<>
    void shuffle4(const SizedVector<4, SSEDouble> &A,
                  const SizedVector<4, SSEDouble> &B,
                  const SizedVector<4, SSEDouble> &C,
                  const SizedVector<4, SSEDouble> &D,
                  SizedVector<4, SSEDouble> *ptr1,
                  SizedVector<4, SSEDouble> *ptr2,
                  SizedVector<4, SSEDouble> *ptr3,
                  SizedVector<4, SSEDouble> *ptr4)
    {
        ptr1->mData[0] = SSEDouble::shuffle<0, 0>(A.mData[0], C.mData[0]);
        ptr1->mData[1] = SSEDouble::shuffle<0, 0>(B.mData[0], D.mData[0]);
        ptr2->mData[0] = SSEDouble::shuffle<0, 0>(A.mData[1], C.mData[1]);
        ptr2->mData[1] = SSEDouble::shuffle<0, 0>(B.mData[1], D.mData[1]);
        ptr3->mData[0] = SSEDouble::shuffle<1, 1>(A.mData[0], C.mData[0]);
        ptr3->mData[1] = SSEDouble::shuffle<1, 1>(B.mData[0], D.mData[0]);
        ptr4->mData[0] = SSEDouble::shuffle<1, 1>(A.mData[1], C.mData[1]);
        ptr4->mData[1] = SSEDouble::shuffle<1, 1>(B.mData[1], D.mData[1]);
    }
    
#endif

    // ******************** Templates (Scalar or SIMD) for FFT Passes ******************** //
    
    // Pass One and Two with Re-ordering
    
    template <class T> void pass_1_2_reorder(Split<typename T::scalar_type> *input, uintptr_t length)
    {
        typedef SizedVector<4, T> Vector;
        
        Vector *r1_ptr = reinterpret_cast<Vector *>(input->realp);
        Vector *r2_ptr = r1_ptr + (length >> 4);
        Vector *r3_ptr = r2_ptr + (length >> 4);
        Vector *r4_ptr = r3_ptr + (length >> 4);
        Vector *i1_ptr = reinterpret_cast<Vector *>(input->imagp);
        Vector *i2_ptr = i1_ptr + (length >> 4);
        Vector *i3_ptr = i2_ptr + (length >> 4);
        Vector *i4_ptr = i3_ptr + (length >> 4);
        
        for (uintptr_t i = 0; i < length >> 4; i++)
        {
            const Vector r1 = *r1_ptr;
            const Vector i1 = *i1_ptr;
            const Vector r2 = *r2_ptr;
            const Vector i2 = *i2_ptr;
            
            const Vector r3 = *r3_ptr;
            const Vector i3 = *i3_ptr;
            const Vector r4 = *r4_ptr;
            const Vector i4 = *i4_ptr;
            
            const Vector r5 = r1 + r3;
            const Vector r6 = r2 + r4;
            const Vector r7 = r1 - r3;
            const Vector r8 = r2 - r4;
            
            const Vector i5 = i1 + i3;
            const Vector i6 = i2 + i4;
            const Vector i7 = i1 - i3;
            const Vector i8 = i2 - i4;
            
            const Vector rA = r5 + r6;
            const Vector rB = r5 - r6;
            const Vector rC = r7 + i8;
            const Vector rD = r7 - i8;
            
            const Vector iA = i5 + i6;
            const Vector iB = i5 - i6;
            const Vector iC = i7 - r8;
            const Vector iD = i7 + r8;
            
            shuffle4(rA, rB, rC, rD, r1_ptr++, r2_ptr++, r3_ptr++, r4_ptr++);
            shuffle4(iA, iB, iC, iD, i1_ptr++, i2_ptr++, i3_ptr++, i4_ptr++);
        }
    }
    
    // Pass Three Twiddle Factors
    
    template <class T> void pass_3_twiddle(SizedVector<4, T> &tr, SizedVector<4, T> &ti)
    {
        static const double SQRT_2_2 = 0.70710678118654752440084436210484904;
        
        typename T::scalar_type _______zero = static_cast<typename T::scalar_type>(0);
        typename T::scalar_type ________one = static_cast<typename T::scalar_type>(1);
        typename T::scalar_type neg_____one = static_cast<typename T::scalar_type>(-1);
        typename T::scalar_type ____sqrt2_2 = static_cast<typename T::scalar_type>(SQRT_2_2);
        typename T::scalar_type neg_sqrt2_2 = static_cast<typename T::scalar_type>(-SQRT_2_2);
        
        typename T::scalar_type str[4] = {________one, ____sqrt2_2, _______zero, neg_sqrt2_2};
        typename T::scalar_type sti[4] = {_______zero, neg_sqrt2_2, neg_____one, neg_sqrt2_2};
        
        tr = SizedVector<4, T>(str);
        ti = SizedVector<4, T>(sti);
    }
    
    // Pass Three With Re-ordering
    
    template <class T> void pass_3_reorder(Split<typename T::scalar_type> *input, uintptr_t length)
    {
        typedef SizedVector<4, T> Vector;
        
        uintptr_t offset = length >> 5;
        uintptr_t outerLoop = length >> 6;
        
        Vector tr;
        Vector ti;
        
        pass_3_twiddle(tr, ti);
        
        Vector *r1_ptr = reinterpret_cast<Vector *>(input->realp);
        Vector *i1_ptr = reinterpret_cast<Vector *>(input->imagp);
        Vector *r2_ptr = r1_ptr + offset;
        Vector *i2_ptr = i1_ptr + offset;
        
        for (uintptr_t i = 0, j = 0; i < length >> 1; i += 8)
        {
            // Get input
            
            const Vector r1(r1_ptr);
            const Vector r2(r1_ptr + 1);
            const Vector i1(i1_ptr);
            const Vector i2(i1_ptr + 1);
            
            const Vector r3(r2_ptr);
            const Vector r4(r2_ptr + 1);
            const Vector i3(i2_ptr);
            const Vector i4(i2_ptr + 1);
            
            // Multiply by twiddle
            
            const Vector r5 = (r3 * tr) - (i3 * ti);
            const Vector i5 = (r3 * ti) + (i3 * tr);
            const Vector r6 = (r4 * tr) - (i4 * ti);
            const Vector i6 = (r4 * ti) + (i4 * tr);
            
            // Store output (swapping as necessary)
            
            *r1_ptr++ = r1 + r5;
            *r1_ptr++ = r1 - r5;
            *i1_ptr++ = i1 + i5;
            *i1_ptr++ = i1 - i5;
            
            *r2_ptr++ = r2 + r6;
            *r2_ptr++ = r2 - r6;
            *i2_ptr++ = i2 + i6;
            *i2_ptr++ = i2 - i6;
            
            if (!(++j % outerLoop))
            {
                r1_ptr += offset;
                r2_ptr += offset;
                i1_ptr += offset;
                i2_ptr += offset;
            }
        }
    }
    
    // Pass Three Without Re-ordering
    
    template <class T> void pass_3(Split<typename T::scalar_type> *input, uintptr_t length)
    {
        typedef SizedVector<4, T> Vector;
        
        Vector tr;
        Vector ti;
        
        pass_3_twiddle(tr, ti);
        
        Vector *r_ptr = reinterpret_cast<Vector *>(input->realp);
        Vector *i_ptr = reinterpret_cast<Vector *>(input->imagp);
        
        for (uintptr_t i = 0; i < length >> 3; i++)
        {
            // Get input
            
            const Vector r1(r_ptr);
            const Vector r2(r_ptr + 1);
            const Vector i1(i_ptr);
            const Vector i2(i_ptr + 1);
            
            // Multiply by twiddle
            
            const Vector r3 = (r2 * tr) - (i2 * ti);
            const Vector i3 = (r2 * ti) + (i2 * tr);
            
            // Store output
            
            *r_ptr++ = r1 + r3;
            *r_ptr++ = r1 - r3;
            *i_ptr++ = i1 + i3;
            *i_ptr++ = i1 - i3;
            
        }
    }
    
    // A Pass Requiring Tables With Re-ordering
    
    template <class T> void pass_trig_table_reorder(typename T::split_type *input, typename T::setup_type *setup, uintptr_t length, uintptr_t pass)
    {
        uintptr_t size = 2 << pass;
        uintptr_t incr = size / (T::size << 1);
        uintptr_t loop = size;
        uintptr_t offset = (length >> pass) / (T::size << 1);
        uintptr_t outerLoop = ((length >> 1) / size) / ((uintptr_t) 1 << pass);
        
        T *r1_ptr = reinterpret_cast<T *>(input->realp);
        T *i1_ptr = reinterpret_cast<T *>(input->imagp);
        T *r2_ptr = r1_ptr + offset;
        T *i2_ptr = i1_ptr + offset;
        
        for (uintptr_t i = 0, j = 0; i < (length >> 1); loop += size)
        {
            T *tr_ptr = reinterpret_cast<T *>(setup->tables[pass - (trig_table_offset - 1)].realp);
            T *ti_ptr = reinterpret_cast<T *>(setup->tables[pass - (trig_table_offset - 1)].imagp);
            
            for (; i < loop; i += (T::size << 1))
            {
                // Get input and twiddle
                
                const T tr = *tr_ptr++;
                const T ti = *ti_ptr++;
                
                const T r1 = *r1_ptr;
                const T i1 = *i1_ptr;
                const T r2 = *r2_ptr;
                const T i2 = *i2_ptr;
                
                const T r3 = *(r1_ptr + incr);
                const T i3 = *(i1_ptr + incr);
                const T r4 = *(r2_ptr + incr);
                const T i4 = *(i2_ptr + incr);
                
                // Multiply by twiddle
                
                const T r5 = (r2 * tr) - (i2 * ti);
                const T i5 = (r2 * ti) + (i2 * tr);
                const T r6 = (r4 * tr) - (i4 * ti);
                const T i6 = (r4 * ti) + (i4 * tr);
                
                // Store output (swapping as necessary)
                
                *r1_ptr = r1 + r5;
                *(r1_ptr++ + incr) = r1 - r5;
                *i1_ptr = i1 + i5;
                *(i1_ptr++ + incr) = i1 - i5;
                
                *r2_ptr = r3 + r6;
                *(r2_ptr++ + incr) = r3 - r6;
                *i2_ptr = i3 + i6;
                *(i2_ptr++ + incr) = i3 - i6;
            }
            
            r1_ptr += incr;
            r2_ptr += incr;
            i1_ptr += incr;
            i2_ptr += incr;
            
            if (!(++j % outerLoop))
            {
                r1_ptr += offset;
                r2_ptr += offset;
                i1_ptr += offset;
                i2_ptr += offset;
            }
        }
    }
    
    // A Pass Requiring Tables Without Re-ordering
    
    template <class T> void pass_trig_table(typename T::split_type *input, typename T::setup_type *setup, uintptr_t length, uintptr_t pass)
    {
        uintptr_t size = 2 << pass;
        uintptr_t incr = size / (T::size << 1);
        uintptr_t loop = size;
        
        T *r1_ptr = reinterpret_cast<T *>(input->realp);
        T *i1_ptr = reinterpret_cast<T *>(input->imagp);
        T *r2_ptr = r1_ptr + (size >> 1) / T::size;
        T *i2_ptr = i1_ptr + (size >> 1) / T::size;
        
        for (uintptr_t i = 0; i < length; loop += size)
        {
            T *tr_ptr = reinterpret_cast<T *>(setup->tables[pass - (trig_table_offset - 1)].realp);
            T *ti_ptr = reinterpret_cast<T *>(setup->tables[pass - (trig_table_offset - 1)].imagp);
            
            for (; i < loop; i += (T::size << 1))
            {
                // Get input and twiddle factors
                
                const T tr = *tr_ptr++;
                const T ti = *ti_ptr++;
                
                const T r1 = *r1_ptr;
                const T i1 = *i1_ptr;
                const T r2 = *r2_ptr;
                const T i2 = *i2_ptr;
                
                // Multiply by twiddle
                
                const T r3 = (r2 * tr) - (i2 * ti);
                const T i3 = (r2 * ti) + (i2 * tr);
                
                // Store output
                
                *r1_ptr++ = r1 + r3;
                *i1_ptr++ = i1 + i3;
                *r2_ptr++ = r1 - r3;
                *i2_ptr++ = i1 - i3;
            }
            
            r1_ptr += incr;
            r2_ptr += incr;
            i1_ptr += incr;
            i2_ptr += incr;
        }
    }
    
    // A Real Pass Requiring Trig Tables (Never Reorders)
    
    template <bool ifft, class T> void pass_real_trig_table(Split<T> *input, Setup<T> *setup, uintptr_t fft_log2)
    {
        uintptr_t length = (uintptr_t) 1 << (fft_log2 - 1);
        uintptr_t lengthM1 = length - 1;
        
        T *r1_ptr = input->realp;
        T *i1_ptr = input->imagp;
        T *r2_ptr = r1_ptr + lengthM1;
        T *i2_ptr = i1_ptr + lengthM1;
        T *tr_ptr = setup->tables[fft_log2 - trig_table_offset].realp;
        T *ti_ptr = setup->tables[fft_log2 - trig_table_offset].imagp;
        
        // Do DC and Nyquist (note that the complex values can be considered periodic)
        
        const T t1 = r1_ptr[0] + i1_ptr[0];
        const T t2 = r1_ptr[0] - i1_ptr[0];
        
        *r1_ptr++ = ifft ? t1 : t1 + t1;
        *i1_ptr++ = ifft ? t2 : t2 + t2;
        
        tr_ptr++;
        ti_ptr++;
        
        // N.B. - The last time through this loop will write the same values twice to the same places
        // N.B. - In this case: t1 == 0, i4 == 0, r1_ptr == r2_ptr, i1_ptr == i2_ptr
        
        for (uintptr_t i = 0; i < (length >> 1); i++)
        {
            const T tr = ifft ? -*tr_ptr++ : *tr_ptr++;
            const T ti = *ti_ptr++;
            
            // Get input
            
            const T r1 = *r1_ptr;
            const T i1 = *i1_ptr;
            const T r2 = *r2_ptr;
            const T i2 = *i2_ptr;
            
            const T r3 = r1 + r2;
            const T i3 = i1 + i2;
            const T r4 = r1 - r2;
            const T i4 = i1 - i2;
            
            const T t1 = (tr * i3) + (ti * r4);
            const T t2 = (ti * i3) - (tr * r4);
            
            // Store output
            
            *r1_ptr++ = r3 + t1;
            *i1_ptr++ = t2 + i4;
            *r2_ptr-- = r3 - t1;
            *i2_ptr-- = t2 - i4;
        }
    }
    
    // ******************** Scalar-Only Small FFTs ******************** //
    
    // Small Complex FFTs (2, 4 or 8 points)
    
    template <class T> void small_fft(Split<T> *input, uintptr_t fft_log2)
    {
        T *r1_ptr = input->realp;
        T *i1_ptr = input->imagp;
        
        if (fft_log2 == 1)
        {
            const T r1 = r1_ptr[0];
            const T r2 = r1_ptr[1];
            const T i1 = i1_ptr[0];
            const T i2 = i1_ptr[1];
            
            r1_ptr[0] = r1 + r2;
            r1_ptr[1] = r1 - r2;
            i1_ptr[0] = i1 + i2;
            i1_ptr[1] = i1 - i2;
        }
        else if (fft_log2 == 2)
        {
            const T r5 = r1_ptr[0];
            const T r6 = r1_ptr[1];
            const T r7 = r1_ptr[2];
            const T r8 = r1_ptr[3];
            const T i5 = i1_ptr[0];
            const T i6 = i1_ptr[1];
            const T i7 = i1_ptr[2];
            const T i8 = i1_ptr[3];
            
            // Pass One
            
            const T r1 = r5 + r7;
            const T r2 = r5 - r7;
            const T r3 = r6 + r8;
            const T r4 = r6 - r8;
            const T i1 = i5 + i7;
            const T i2 = i5 - i7;
            const T i3 = i6 + i8;
            const T i4 = i6 - i8;
            
            // Pass Two
            
            r1_ptr[0] = r1 + r3;
            r1_ptr[1] = r2 + i4;
            r1_ptr[2] = r1 - r3;
            r1_ptr[3] = r2 - i4;
            i1_ptr[0] = i1 + i3;
            i1_ptr[1] = i2 - r4;
            i1_ptr[2] = i1 - i3;
            i1_ptr[3] = i2 + r4;
        }
        else if (fft_log2 == 3)
        {
            // Pass One
            
            const T r1 = r1_ptr[0] + r1_ptr[4];
            const T r2 = r1_ptr[0] - r1_ptr[4];
            const T r3 = r1_ptr[2] + r1_ptr[6];
            const T r4 = r1_ptr[2] - r1_ptr[6];
            const T r5 = r1_ptr[1] + r1_ptr[5];
            const T r6 = r1_ptr[1] - r1_ptr[5];
            const T r7 = r1_ptr[3] + r1_ptr[7];
            const T r8 = r1_ptr[3] - r1_ptr[7];
            
            const T i1 = i1_ptr[0] + i1_ptr[4];
            const T i2 = i1_ptr[0] - i1_ptr[4];
            const T i3 = i1_ptr[2] + i1_ptr[6];
            const T i4 = i1_ptr[2] - i1_ptr[6];
            const T i5 = i1_ptr[1] + i1_ptr[5];
            const T i6 = i1_ptr[1] - i1_ptr[5];
            const T i7 = i1_ptr[3] + i1_ptr[7];
            const T i8 = i1_ptr[3] - i1_ptr[7];
            
            // Pass Two
            
            r1_ptr[0] = r1 + r3;
            r1_ptr[1] = r2 + i4;
            r1_ptr[2] = r1 - r3;
            r1_ptr[3] = r2 - i4;
            r1_ptr[4] = r5 + r7;
            r1_ptr[5] = r6 + i8;
            r1_ptr[6] = r5 - r7;
            r1_ptr[7] = r6 - i8;
            
            i1_ptr[0] = i1 + i3;
            i1_ptr[1] = i2 - r4;
            i1_ptr[2] = i1 - i3;
            i1_ptr[3] = i2 + r4;
            i1_ptr[4] = i5 + i7;
            i1_ptr[5] = i6 - r8;
            i1_ptr[6] = i5 - i7;
            i1_ptr[7] = i6 + r8;
            
            // Pass Three
            
            pass_3<Scalar<T> >(input, 8);
        }
    }
    
    // Small Real FFTs (2 or 4 points)
    
    template <bool ifft, class T> void small_real_fft(Split<T> *input, uintptr_t fft_log2)
    {
        T *r1_ptr = input->realp;
        T *i1_ptr = input->imagp;
        
        if (fft_log2 == 1)
        {
            const T r1 = ifft ? r1_ptr[0] : r1_ptr[0] + r1_ptr[0];
            const T r2 = ifft ? i1_ptr[0] : i1_ptr[0] + i1_ptr[0];
            
            r1_ptr[0] = (r1 + r2);
            i1_ptr[0] = (r1 - r2);
        }
        else if (fft_log2 == 2)
        {
            if (!ifft)
            {
                // Pass One
                
                const T r1 = r1_ptr[0] + r1_ptr[1];
                const T r2 = r1_ptr[0] - r1_ptr[1];
                const T i1 = i1_ptr[0] + i1_ptr[1];
                const T i2 = i1_ptr[1] - i1_ptr[0];
                
                // Pass Two
                
                const T r3 = r1 + i1;
                const T i3 = r1 - i1;
                
                r1_ptr[0] = r3 + r3;
                r1_ptr[1] = r2 + r2;
                i1_ptr[0] = i3 + i3;
                i1_ptr[1] = i2 + i2;
            }
            else
            {
                const T i1 = r1_ptr[0];
                const T r2 = r1_ptr[1] + r1_ptr[1];
                const T i2 = i1_ptr[0];
                const T r4 = i1_ptr[1] + i1_ptr[1];
                
                // Pass One
                
                const T r1 = i1 + i2;
                const T r3 = i1 - i2;
                
                // Pass Two
                
                r1_ptr[0] = r1 + r2;
                r1_ptr[1] = r1 - r2;
                i1_ptr[0] = r3 - r4;
                i1_ptr[1] = r3 + r4;
            }
        }
    }
    
    // ******************** Unzip and Zip ******************** //
    
    // Unzip
    
    template <class T, class U, class V> void unzip_complex(const U *input, V *output, uintptr_t half_length)
    {
        T *realp = output->realp;
        T *imagp = output->imagp;
        
        for (uintptr_t i = 0; i < half_length; i++)
        {
            *realp++ = static_cast<T>(*input++);
            *imagp++ = static_cast<T>(*input++);
        }
    }
    
    // Zip
    
    template <class T, class U> void zip_complex(const T *input, U *output, uintptr_t half_length)
    {
        U *realp = input->realp;
        U *imagp = input->imagp;
        
        for (uintptr_t i = 0; i < half_length; i++)
        {
            *output++ = *realp++;
            *output++ = *imagp++;
        }
    }
    
    // Unzip With Zero Padding
    
    template <class T, class U, class V> void unzip_zero(const U *input, V *output, uintptr_t in_length, uintptr_t log2n)
    {
        T odd_sample = static_cast<T>(input[in_length - 1]);
        T *realp = output->realp;
        T *imagp = output->imagp;
        
        // Check input length is not longer than the FFT size and unzip an even number of samples
        
        uintptr_t fft_size = static_cast<uintptr_t>(1 << log2n);
        in_length = std::min(fft_size, in_length);
        unzip_complex<T>(input, output, in_length >> 1);
        
        // If necessary replace the odd sample, and zero pad the input
        
        if (fft_size > in_length)
        {
            uintptr_t end_point1 = in_length >> 1;
            uintptr_t end_point2 = fft_size >> 1;
            
            realp[end_point1] = (in_length & 1) ? odd_sample : static_cast<T>(0);
            imagp[end_point1] = static_cast<T>(0);
            
            for (uintptr_t i = end_point1 + 1; i < end_point2; i++)
            {
                realp[i] = static_cast<T>(0);
                imagp[i] = static_cast<T>(0);
            }
        }
    }
    
    // ******************** FFT Pass Control ******************** //
    
    // FFT Passes Template
    
    template <class T, class U, class V, class W, class X> void fft_passes(Split<X> *input, Setup<X> *setup, uintptr_t fft_log2)
    {
        const uintptr_t length = (uintptr_t) 1 << fft_log2;
        uintptr_t i;
        
        pass_1_2_reorder<T>(input, length);
        
        if (fft_log2 > 5)
            pass_3_reorder<U>(input, length);
        else
            pass_3<U>(input, length);
        
        if (3 < (fft_log2 >> 1))
            pass_trig_table_reorder<V>(input, setup, length, 3);
        else
            pass_trig_table<V>(input, setup, length, 3);

        for (i = 4; i < (fft_log2 >> 1); i++)
            pass_trig_table_reorder<W>(input, setup, length, i);
        
        for (; i < fft_log2; i++)
            pass_trig_table<W>(input, setup, length, i);
    }
    
    // SIMD Options
    
    template <class T>
    void fft_passes_simd(Split<T> *input, Setup<T> *setup, uintptr_t fft_log2)
    {
        fft_passes<Scalar<T>, Scalar<T>, Scalar<T>, Scalar<T> >(input, setup, fft_log2);
    }
    
#if (SIMD_COMPILER_SUPPORT_LEVEL == SIMD_COMPILER_SUPPORT_SSE)
    
    // SIMD Double Specialisation
    
    template<> void fft_passes_simd(Split<double> *input, Setup<double> *setup, uintptr_t fft_log2)
    {
        fft_passes<SSEDouble, SSEDouble, SSEDouble, SSEDouble>(input, setup, fft_log2);
    }
    
    // SIMD Float Specialisation
    
    template<> void fft_passes_simd(Split<float> *input, Setup<float> *setup, uintptr_t fft_log2)
    {
        fft_passes<SSEFloat, SSEFloat, SSEFloat, SSEFloat>(input, setup, fft_log2);
    }

#endif
    
#if (SIMD_COMPILER_SUPPORT_LEVEL == SIMD_COMPILER_SUPPORT_AVX256)

    // SIMD Double Specialisation
    
    template<>
    void fft_passes_simd(Split<double> *input, Setup<double> *setup, uintptr_t fft_log2)
    {
        if (SIMD_Support >= kAVX256)
            fft_passes<SSEDouble, AVX256Double, AVX256Double, AVX256Double>(input, setup, fft_log2);
        else
            fft_passes<SSEDouble, SSEDouble, SSEDouble, SSEDouble>(input, setup, fft_log2);
    }
    
    // SIMD Float Specialisation
    
    template<>
    void fft_passes_simd(Split<float> *input, Setup<float> *setup, uintptr_t fft_log2)
    {
        if (SIMD_Support >= kAVX256)
            fft_passes<SSEFloat, SSEFloat, AVX256Float, AVX256Float>(input, setup, fft_log2);
        else
            fft_passes<SSEFloat, SSEFloat, SSEFloat, SSEFloat>(input, setup, fft_log2);
    }

#endif
    
#if (SIMD_COMPILER_SUPPORT_LEVEL == SIMD_COMPILER_SUPPORT_AVX512)
    
    // SIMD Double Specialisation
    
    template<>
    void fft_passes_simd(Split<double> *input, Setup<double> *setup, uintptr_t fft_log2)
    {
        if (SIMD_Support >= kAVX512)
            fft_passes<SSEDouble, AVX256Double, AVX512Double, AVX512Double>(input, setup, fft_log2);
        else if (SIMD_Support >= kAVX256)
            fft_passes<SSEDouble, AVX256Double, AVX256Double, AVX256Double>(input, setup, fft_log2);
        else
            fft_passes<SSEDouble, SSEDouble, SSEDouble, SSEDouble>(input, setup, fft_log2);
    }
    
    // SIMD Float Specialisation
    
    template<>
    void fft_passes_simd(Split<float> *input, Setup<float> *setup, uintptr_t fft_log2)
    {
        if (SIMD_Support >= kAVX512)
            fft_passes<SSEFloat, SSEFloat, AVX256Float, AVX512Float>(input, setup, fft_log2);
        else if (SIMD_Support >= kAVX256)
            fft_passes<SSEFloat, SSEFloat, AVX256Float, AVX256Float>(input, setup, fft_log2);
        else
            fft_passes<SSEFloat, SSEFloat, SSEFloat, SSEFloat>(input, setup, fft_log2);
    }
    
#endif

    // ******************** Main Calls ******************** //
    
    // A Complex FFT
    
    template <class T>void hisstools_fft(Split<T> *input, Setup<T> *setup, uintptr_t fft_log2)
    {
        if (fft_log2 >= 4)
        {
            if (reinterpret_cast<uintptr_t>(input->realp) % 16 || reinterpret_cast<uintptr_t>(input->imagp) % 16 || SIMD_Support == kNone)
                fft_passes<Scalar<T>, Scalar<T>, Scalar<T>, Scalar<T> >(input, setup, fft_log2);
            else
                fft_passes_simd(input, setup, fft_log2);
        }
        else
            small_fft(input, fft_log2);
    }
    
    // A Complex iFFT
    
    template <class T>void hisstools_ifft(Split<T> *input, Setup<T> *setup, uintptr_t fft_log2)
    {
        Split<T> swap(input->imagp, input->realp);
        hisstools_fft(&swap, setup, fft_log2);
    }
    
    // A Real FFT
    
    template <class T>void hisstools_rfft(Split<T> *input, Setup<T> *setup, uintptr_t fft_log2)
    {
        if (fft_log2 >= 3)
        {
            hisstools_fft(input, setup, fft_log2 - 1);
            pass_real_trig_table<false>(input, setup, fft_log2);
        }
        else
            small_real_fft<false>(input, fft_log2);
    }
    
    // A Real iFFT
    
    template <class T>void hisstools_rifft(Split<T> *input, Setup<T> *setup, uintptr_t fft_log2)
    {
        if (fft_log2 >= 3)
        {
            pass_real_trig_table<true>(input, setup, fft_log2);
            hisstools_ifft(input, setup, fft_log2 - 1);
        }
        else
            small_real_fft<true>(input, fft_log2);
    }
    
} /* hisstools_fft_impl */
