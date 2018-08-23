#pragma once

#include <vector>
#include <iterator>
#include "data/FluidTensor.hpp"
#include "algorithms/NMF.hpp"
#include "algorithms/STFT.hpp"
#include "algorithms/RatioMask.hpp"

using fluid::FluidTensor;
using fluid::nmf::NMF;
using fluid::stft::STFT;
using fluid::stft::ISTFT;
using fluid::stft::Spectrogram;

namespace fluid {
  namespace nmf{
    
    /**
     Integration class for doing NMF filtering and resynthesis
     **/
     
    class NMFClient
    {
      
      //        using vec_iterator = std::vector<double>::const_iterator;
      //        using source_iterator = std::vector<std::vector<double>>::const_iterator;
      
    public:
      //No, you may not construct an empty instance, or copy this, or move this
      NMFClient() = delete;
      NMFClient(NMFClient&)=delete;
      NMFClient(NMFClient&&)=delete;
      NMFClient operator=(NMFClient&)=delete;
      NMFClient operator=(NMFClient&&)=delete;
      
      /**
       You may constrct one by supplying some senisble numbers here
       rank: NMF rank
       iterations: max nmf iterations
       fft_size: power 2 pls
       **/
        NMFClient(size_t rank, size_t iterations, size_t fft_size, size_t window_size, size_t hop_size):
        m_rank(rank),m_iterations(iterations), m_window_size(window_size),m_fft_size(fft_size),m_hop_size(hop_size)
      {}
      
      ~NMFClient()= default;
      //Not implemented
      //void reset();
      //bool isReady() const;
      
      /***
       Take some data, NMF it
       ***/
      void process(const FluidTensor<double,1> &data, bool resynthesise)
      {
        m_audio_buffers.resize(m_rank,data.extent(0));
        
        m_has_processed = false;
        m_has_resynthed = false;
        STFT stft(m_window_size,m_fft_size,m_hop_size);
        Spectrogram spec = stft.process(data);
        FluidTensor<double, 2> mag = spec.getMagnitude();
        NMF nmf(m_rank,m_iterations);
        m_model = nmf.process(spec.getMagnitude());
        m_has_processed = true;
        
        if(resynthesise)
        {
          ratiomask::RatioMask mask(m_model.getMixEstimate(),1);
          ISTFT istft(m_window_size, m_fft_size, m_hop_size);
          for(int i = 0; i < m_rank; ++i)
          {
            RealMatrix estimate = m_model.getEstimate(i);
            Spectrogram result(mask.process(spec.mData, estimate));
            RealVector audio = istft.process(result);
              m_audio_buffers.row(i) = audio(fluid::slice(0,data.extent(0)));
          }
          m_has_resynthed = true;
        }
      }
      
      /***
       Report the size of a dictionary, in bins (= fft_size/2)
       ***/
      size_t dictionary_size() const
      {
        return m_has_processed ? m_model.getW().extent(0) : 0 ;
      }
      
      /***
       Report the length of an activation, in frames
       ***/
      size_t activations_length() const{
        return m_has_processed ? m_model.getH().extent(1) : 0;
      }
      
      /***
       Report the number of sources (i.e. the rank
       ***/
      size_t num_sources() const
      {
        return m_has_resynthed ? m_audio_buffers.size() : 0;
      }
      //        size_t rank() const;
      
      /***
       Retreive the dictionary at the given index
       ***/
      const FluidTensorView<double, 1> dictionary(const size_t idx) const
      {
        assert(m_has_processed && idx < m_model.W.cols());
        return m_model.getW().col(idx);
      }
      
      /***
       Retreive the activation at the given index
       ***/
      const FluidTensorView<double, 1> activation(const size_t idx) const
      {
        assert(m_has_processed && idx < m_model.H.rows());
        return m_model.getH().row(idx);
      }
      
      /***
       Retreive the resynthesized source at the given index (so long as resyntheiss has happened, mind
       ***/
      FluidTensorView<const double, 1> source(const size_t idx) const
      {
        assert(idx < m_audio_buffers.rows() && "Range Error");
        return m_audio_buffers.row(idx);
      }
      
      //        source_iterator sources_begin() const ;
      //        source_iterator sources_end()const;
      
      /***
       Get the whole of dictionaries / activations as a 2D structure
       ***/
      FluidTensor<double,2> dictionaries() const
      {
        return m_model.getW();
      }
      FluidTensor<double,2> activations() const
      {
        return m_model.getH();
      }
    private:
      size_t m_rank;
      size_t m_iterations;      
      size_t m_window_size;
      size_t m_fft_size;
      size_t m_hop_size;
      bool m_has_processed;
      bool m_has_resynthed;
      fluid::nmf::NMFModel m_model;
      FluidTensor<double,2> m_audio_buffers;
    };
  } //namespace max
} //namesapce fluid
