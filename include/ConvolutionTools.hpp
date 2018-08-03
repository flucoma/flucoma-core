
#pragma once

#include <SIMDSupport.hpp>
#include <HISSTools_FFT/HISSTools_FFT.h>
#include <vector>

namespace fluid {
namespace convolution {

// The edge mode determines wraparound etc.
  
enum EdgeMode { kEdgeLinear, kEdgeWrap, kEdgeWrapCentre, kEdgeFold };

    
namespace impl {  // Here is the underlying implementation
                  // See the bottom of the file for interface
                  // Note that the function calls will currently allocate and dellocate on the heap

  // Setups
  
  struct FFTComplexSetup
  {
    FFTComplexSetup(size_t maxFFTLog2) { hisstools_create_setup(&mSetup, maxFFTLog2); }
    ~FFTComplexSetup() { hisstools_destroy_setup(mSetup); }
    
    FFTComplexSetup(const FFTComplexSetup&) = delete;
    FFTComplexSetup operator =(const FFTComplexSetup&) = delete;
    
    FFT_SETUP_D mSetup;
  };
  
  struct FFTRealSetup : public FFTComplexSetup
  {
    FFTRealSetup(size_t maxFFTLog2) : FFTComplexSetup(maxFFTLog2 - 1) {};
  };
  
  // Temporary Memory
  
  struct TempSpectra
  {
    TempSpectra(size_t dataSize)
    {
      mData.resize(dataSize * 2);
      mSpectra.realp = mData.data();
      mSpectra.imagp = mSpectra.realp + dataSize;
    }
    
    FFT_SPLIT_COMPLEX_D mSpectra;
    std::vector<double> mData;
  };
  
  struct ConvolveOp
  {
    template<class T>
    void operator()(T &outR, T &outI, const T a, const T b, const T c, const T d, T scale)
    {
      outR = scale * (a * c - b * d);
      outI = scale * (a * d + b * c);
    }
  };
  
  struct CorrelateOp
  {
    template<class T>
    void operator()(T &outR, T &outI, const T a, const T b, const T c, const T d, T scale)
    {
      outR = scale * (a * c + b * d);
      outI = scale * (b * c - a * d);
    }
  };
  
  // Complex Forward Transform
  
  void transformForward(FFTComplexSetup& setup, FFT_SPLIT_COMPLEX_D &io, size_t FFTSizelog2)
  {
    hisstools_fft(setup.mSetup, &io, FFTSizelog2);
  }
  
  // Complex Inverse Transform
  
  void transformInverse(FFTComplexSetup& setup, FFT_SPLIT_COMPLEX_D &io, size_t FFTSizelog2)
  {
    hisstools_ifft(setup.mSetup, &io, FFTSizelog2);
  }
  
  // Real Forward Transform (in-place)
  
  void transformForwardReal(FFTRealSetup& setup, FFT_SPLIT_COMPLEX_D &io, size_t FFTSizelog2)
  {
    hisstools_rfft(setup.mSetup, &io, FFTSizelog2);
  }
  
  // Real Forward Tansform (with unzipping)
  
  void transformForwardReal(FFTRealSetup& setup, FFT_SPLIT_COMPLEX_D &output, const double *input, size_t size, size_t FFTSizelog2)
  {
    hisstools_rfft(setup.mSetup, input, &output, size, FFTSizelog2);
  }
  
  // Real Inverse Transform (in-place)
  
  void transformInverseReal(FFTRealSetup& setup, FFT_SPLIT_COMPLEX_D &io, size_t FFTSizelog2)
  {
    hisstools_rifft(setup.mSetup, &io, FFTSizelog2);
  }
  
  // Real Inverse Transform (with zipping)
  
  void transformInverseReal(FFTRealSetup& setup, double *output, FFT_SPLIT_COMPLEX_D &input, size_t FFTSizelog2)
  {
    hisstools_rifft(setup.mSetup, &input, output, FFTSizelog2);
  }
  
  // Size calculations
  
  size_t ilog2(size_t value)
  {
    size_t bitShift = value;
    size_t bitCount = 0;
    
    while (bitShift)
    {
      bitShift >>= 1U;
      bitCount++;
    }
    
    if (bitCount && value == 1U << (bitCount - 1U))
      return bitCount - 1U;
    else
      return bitCount;
  };
  
  size_t calcLinearSize(size_t size1, size_t size2)
  {
    return ( size1 && size2) ? size1 + size2 - 1 : 0;
  }
  
  size_t calcSize(size_t size1, size_t size2, EdgeMode mode)
  {
    size_t linearSize = size1 + size2 - 1;
    size_t sizeOut = mode != kEdgeLinear ? std::max(size1, size2) : linearSize;
    
    if (mode == kEdgeFold && !(std::min(size1, size2) & 1U))
      sizeOut++;
    
    return (!size1 || !size2) ? 0 : sizeOut;
  }
  
  // Arrangement
  
  // Memory manipulation (complex)
  
  void copyZero(double *output, const double *input, size_t inSize, size_t outSize)
  {
    std::copy(input, input + inSize, output);
    std::fill_n(output + inSize, outSize - inSize, 0.0);
  }
  
  void copy(FFT_SPLIT_COMPLEX_D output, const FFT_SPLIT_COMPLEX_D spectrum, size_t outOffset, size_t offset, size_t size)
  {
    std::copy(spectrum.realp + offset, spectrum.realp + size + offset, output.realp + outOffset);
    std::copy(spectrum.imagp + offset, spectrum.imagp + size + offset, output.imagp + outOffset);
  }
  
  void wrap(FFT_SPLIT_COMPLEX_D output, const FFT_SPLIT_COMPLEX_D spectrum, size_t outOffset, size_t offset, size_t size)
  {
    for (size_t i = 0; i < size; i++)
    {
      output.realp[i + outOffset] += spectrum.realp[i + offset];
      output.imagp[i + outOffset] += spectrum.imagp[i + offset];
    }
  }
  
  void fold(FFT_SPLIT_COMPLEX_D output, const FFT_SPLIT_COMPLEX_D spectrum, size_t outOffset, size_t endOffset, size_t size)
  {
    for (size_t i = 1; i <= size; i++)
    {
      output.realp[i + outOffset - 1] += spectrum.realp[endOffset - i];
      output.imagp[i + outOffset - 1] += spectrum.imagp[endOffset - i];
    }
  }

  // Memory manipulation (real)
  
  struct assign { void operator()(double *output, const double *ptr) { *output = *ptr; } };
  struct accum { void operator()(double *output, const double *ptr) { *output += *ptr; } };
  struct increment { template<class T> T *operator()(T *&ptr) { return ptr++; } };
  struct decrement { template<class T> T *operator()(T *&ptr) { return ptr--; } };
  
  template <typename Op, typename PtrOp>
  void zip(double *output, const double *p1, const double *p2, size_t size, Op op, PtrOp pOp)
  {
    for (size_t i = 0; i < (size >> 1); i++)
    {
      op(pOp(output), pOp(p1));
      op(pOp(output), pOp(p2));
    }
    
    if (size & 1U)
      op(pOp(output), pOp(p1));
  }
  
  void copy(double *output, const FFT_SPLIT_COMPLEX_D spectrum, size_t outOffset, size_t offset, size_t size)
  {
    const double *p1 = (offset & 1U) ? spectrum.imagp + (offset >> 1) : spectrum.realp + (offset >> 1);
    const double *p2 = (offset & 1U) ? spectrum.realp + (offset >> 1) + 1 : spectrum.imagp + (offset >> 1);
    
    zip(output + outOffset, p1, p2, size, assign(), increment());
  }
  
  void wrap(double *output, const FFT_SPLIT_COMPLEX_D spectrum, size_t outOffset, size_t offset, size_t size)
  {
    const double *p1 = (offset & 1U) ? spectrum.imagp + (offset >> 1) : spectrum.realp + (offset >> 1);
    const double *p2 = (offset & 1U) ? spectrum.realp + (offset >> 1) + 1 : spectrum.imagp + (offset >> 1);
    
    zip(output + outOffset, p1, p2, size, accum(), increment());
  }
  
  void fold(double *output, const FFT_SPLIT_COMPLEX_D spectrum, size_t outOffset, size_t endOffset, size_t size)
  {
    const double *p1 = (endOffset & 1U) ? spectrum.realp + (endOffset >> 1) : spectrum.imagp + (endOffset >> 1) - 1;
    const double *p2 = (endOffset & 1U) ? spectrum.imagp + (endOffset >> 1) - 1 : spectrum.realp + (endOffset >> 1) - 1;
    
    zip(output + outOffset, p1, p2, size, accum(), decrement());
  }
  
  template <class T>
  void arrangeOutput(T output, FFT_SPLIT_COMPLEX_D spectrum, size_t minSize, size_t sizeOut, size_t linearSize, size_t FFTSize, EdgeMode mode, ConvolveOp op)
  {
    size_t offset = (mode == kEdgeFold || mode == kEdgeWrapCentre) ? (minSize - 1) / 2: 0;
    
    copy(output, spectrum, 0, offset, sizeOut);
    
    if (mode == kEdgeWrap)
      wrap(output, spectrum, 0, sizeOut, linearSize - sizeOut);
    
    if (mode == kEdgeWrapCentre)
    {
      size_t endWrap = (minSize - 1) - offset;
      
      wrap(output, spectrum, 0, linearSize - endWrap, endWrap);
      wrap(output, spectrum, sizeOut - offset, 0, offset);
    }
    
    if (mode == kEdgeFold)
    {
      size_t foldSize = minSize / 2;
      size_t foldOffset = sizeOut - foldSize;
      
      fold(output, spectrum, 0, foldSize, foldSize);
      fold(output, spectrum, foldOffset, linearSize, foldSize);
    }
  }
  
  template <class T>
  void arrangeOutput(T output, FFT_SPLIT_COMPLEX_D spectrum, size_t minSize, size_t sizeOut, size_t linearSize, size_t FFTSize, EdgeMode mode, CorrelateOp op)
  {
    size_t maxSize = (linearSize - minSize) + 1;
    
    if (mode == kEdgeLinear || mode == kEdgeWrap)
    {
      size_t extraSize = minSize - 1;
      
      copy(output, spectrum, 0, 0, maxSize);
      
      if (mode == kEdgeLinear)
        copy(output, spectrum, maxSize, FFTSize - extraSize, extraSize);
      else
        wrap(output, spectrum, (linearSize - (2 * (minSize - 1))), FFTSize - extraSize, extraSize);
    }
    else
    {
      size_t offset = minSize / 2;
      
      copy(output, spectrum, 0, FFTSize - offset, offset);
      copy(output, spectrum, offset, 0, sizeOut - offset);
      
      if (mode == kEdgeWrapCentre)
      {
        size_t endWrap = minSize - offset - 1;
        
        wrap(output, spectrum, 0, maxSize - offset, offset);
        wrap(output, spectrum, sizeOut - endWrap, FFTSize - (minSize - 1), endWrap);
      }
      else
      {
        fold(output, spectrum, 0, FFTSize - (offset - 1), offset);
        fold(output, spectrum, sizeOut - offset, maxSize, offset);
      }
    }
  }
  
  // Calculations
  
  template<typename Op>
  void binaryOp(FFT_SPLIT_COMPLEX_D &io1, FFT_SPLIT_COMPLEX_D &in2, size_t dataLength, double scale, Op op)
  {
    const int vec_size = SIMDLimits<double>::max_size;
    
    if (dataLength == 1 || dataLength < (vec_size / 2))
    {
      op(io1.realp[0], io1.imagp[0], io1.realp[0], io1.imagp[0], in2.realp[0], in2.imagp[0], scale);
    }
    else if (dataLength < vec_size)
    {
      const int current_vec_size = SIMDLimits<double>::max_size / 2;
      
      SIMDType<double, current_vec_size> *real1 = reinterpret_cast<SIMDType<double, current_vec_size> *>(io1.realp);
      SIMDType<double, current_vec_size> *imag1 = reinterpret_cast<SIMDType<double, current_vec_size> *>(io1.imagp);
      SIMDType<double, current_vec_size> *real2 = reinterpret_cast<SIMDType<double, current_vec_size> *>(in2.realp);
      SIMDType<double, current_vec_size> *imag2 = reinterpret_cast<SIMDType<double, current_vec_size> *>(in2.imagp);
      
      SIMDType<double, current_vec_size> scaleVec(scale);
      
      for (size_t i = 0; i < (dataLength / current_vec_size); i++)
        op(real1[i], imag1[i], real1[i], imag1[i], real2[i], imag2[i], scaleVec);
    }
    else
    {
      SIMDType<double, vec_size> *real1 = reinterpret_cast<SIMDType<double, vec_size> *>(io1.realp);
      SIMDType<double, vec_size> *imag1 = reinterpret_cast<SIMDType<double, vec_size> *>(io1.imagp);
      SIMDType<double, vec_size> *real2 = reinterpret_cast<SIMDType<double, vec_size> *>(in2.realp);
      SIMDType<double, vec_size> *imag2 = reinterpret_cast<SIMDType<double, vec_size> *>(in2.imagp);
      
      SIMDType<double, vec_size> scaleVec(scale);
      
      for (size_t i = 0; i < (dataLength / vec_size); i++)
        op(real1[i], imag1[i], real1[i], imag1[i], real2[i], imag2[i], scaleVec);
    }
  }
  
  template<typename Op>
  void binaryOpReal(FFT_SPLIT_COMPLEX_D &spectrum1, FFT_SPLIT_COMPLEX_D &spectrum2, size_t dataLength, double scale, Op op)
  {
    // Store DC and Nyquist Results
    
    const double DC = spectrum1.realp[0] * spectrum2.realp[0] * scale;
    const double Nyquist = spectrum1.imagp[0] * spectrum2.imagp[0] * scale;
    
    binaryOp(spectrum1, spectrum2, dataLength, scale, op);
    
    // Set DC and Nyquist bins
    
    spectrum1.realp[0] = DC;
    spectrum1.imagp[0] = Nyquist;
  }
  
  template<typename Op>
  void binarySpectralOperation(double *rOut, double *iOut, const double *rIn1, size_t sizeR1, const double *iIn1, size_t sizeI1,
                               const double *rIn2, size_t sizeR2, const double *iIn2, size_t sizeI2, EdgeMode mode,  Op op)
  {
    size_t size1 = std::max(sizeR1, sizeI1);
    size_t size2 = std::max(sizeR2, sizeI2);
    
    size_t linearSize = calcLinearSize(size1, size2);
    size_t sizeOut = calcSize(size1, size2, mode);
    size_t FFTSizelog2 = ilog2(linearSize);
    size_t FFTSize = 1 << FFTSizelog2;
    
    FFTComplexSetup setup(FFTSizelog2);

    // Special cases for short inputs
    
    if (!sizeOut)
      return;
    
    if (sizeOut == 1)
    {
      ConvolveOp()(rOut[0], iOut[0], rIn1[0], iIn1[0], rIn2[0], iIn2[0], 1.0);
      return;
    }
    
    // Assign temporary memory
    
    TempSpectra spectrum1(FFTSize);
    TempSpectra spectrum2(FFTSize);
    
    FFT_SPLIT_COMPLEX_D output;
    output.realp = rOut;
    output.imagp = iOut;
    
    // Copy to the inputs
    
    copyZero(spectrum1.mSpectra.realp, rIn1, sizeR1, FFTSize);
    copyZero(spectrum1.mSpectra.imagp, iIn1, sizeI1, FFTSize);
    copyZero(spectrum2.mSpectra.realp, rIn2, sizeR2, FFTSize);
    copyZero(spectrum2.mSpectra.imagp, iIn2, sizeI2, FFTSize);
    
    // Take the Forward FFTs
    
    transformForward(setup, spectrum1.mSpectra, FFTSizelog2);
    transformForward(setup, spectrum2.mSpectra, FFTSizelog2);
    
    // Operate
    
    double scale = 1.0 / (double) FFTSize;
    binaryOp(spectrum1, spectrum2, FFTSize, scale, Op());
    
    // Inverse iFFT
    
    transformInverse(setup, spectrum1.mSpectra, FFTSizelog2);
    arrangeOutput(output, spectrum1, std::min(size1, size2), sizeOut, linearSize, FFTSize, mode, op);
  }
  
  template<typename Op>
  void binarySpectralOperationReal(double *output, const double *in1, size_t size1, const double *in2, size_t size2, EdgeMode mode, Op op)
  {
    size_t linearSize = calcLinearSize(size1, size2);
    size_t sizeOut = calcSize(size1, size2, mode);
    size_t FFTSizelog2 = ilog2(linearSize);
    size_t FFTSize = 1 << FFTSizelog2;
    
    FFTRealSetup setup(FFTSizelog2);
    
    // Special cases for short inputs
    
    if (!sizeOut)
      return;
    
    if (sizeOut == 1)
    {
      output[0] = in1[0] * in2[0];
      return;
    }
    
    // Assign temporary memory
    
    TempSpectra spectrum1(FFTSize >> 1);
    TempSpectra spectrum2(FFTSize >> 1);
    
    // Take the Forward Real FFTs
    
    transformForwardReal(setup, spectrum1.mSpectra, in1, size1, FFTSizelog2);
    transformForwardReal(setup, spectrum2.mSpectra, in2, size2, FFTSizelog2);
    
    // Operate
    
    double scale = 0.25 / (double) FFTSize;
    binaryOpReal(spectrum1.mSpectra, spectrum2.mSpectra, FFTSize >> 1, scale, Op());
    
    // Inverse iFFT
    
    transformInverseReal(setup, spectrum1.mSpectra, FFTSizelog2);
    arrangeOutput(output, spectrum1.mSpectra, std::min(size1, size2), sizeOut, linearSize, FFTSize, mode, op);
  }
}
  
// Interface
  
// Convolution (Real)
  
static void convolveReal(double *output, const double *in1, size_t size1, const double *in2, size_t size2, EdgeMode mode = kEdgeWrap)
{
  impl::binarySpectralOperationReal(output, in1, size1, in2, size2, mode, impl::ConvolveOp());
}

// Correlation (Real)

static void correlateReal(double *output, const double *in1, size_t size1, const double *in2, size_t size2, EdgeMode mode = kEdgeWrap)
{
  impl::binarySpectralOperationReal(output, in1, size1, in2, size2, mode, impl::CorrelateOp());
}

// Autocorrelation (Real) (inefficient for now)

static void autocorrelateReal(double *output, const double *in, size_t size, EdgeMode mode = kEdgeWrap)
{
    correlateReal(output, in, size, in, size, mode);
}
  
// Convolution (Complex)

static void convolve(double *rOut, double *iOut, const double *rIn1, size_t sizeR1, const double *iIn1, size_t sizeI1, const double *rIn2, size_t sizeR2, const double *iIn2, size_t sizeI2, EdgeMode mode = kEdgeWrap)
  {
    impl::binarySpectralOperation(rOut, iOut, rIn1, sizeR1, iIn1, sizeI1, rIn2, sizeR2, iIn2, sizeI2, mode, impl::ConvolveOp());
  }
  
  // Correlation (Complex)

  static void correlate(double *rOut, double *iOut, const double *rIn1, size_t sizeR1, const double *iIn1, size_t sizeI1, const double *rIn2, size_t sizeR2, const double *iIn2, size_t sizeI2, EdgeMode mode = kEdgeWrap)
{
    impl::binarySpectralOperation(rOut, iOut, rIn1, sizeR1, iIn1, sizeI1, rIn2, sizeR2, iIn2, sizeI2, mode, impl::CorrelateOp());
}
  
  // Autocorrelation (Complex) (inefficient for now)

static void autocorrelate(double *rOut, double *iOut, const double *rIn, size_t sizeR, const double *iIn, size_t sizeI, EdgeMode mode = kEdgeWrap)
{
    correlate(rOut, iOut, rIn, sizeR, iIn, sizeI, rIn, sizeR, iIn, sizeI, mode);
}

} // namespace convolution
} // namespace fluid

