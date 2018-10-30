#include <cmath>

#include "algorithms/Windows.hpp"
#include "data/FluidBuffers.hpp"

using fluid::FluidSource;
using fluid::FluidTensor;
using fluid::Slice;

int main(int argc, char *argv[]) {
  //    using fluid::FluidSource;
  //    using fluid::FluidSink;
  //
  //    FluidSource<double> source(100);
  //
  //    FluidTensor<double, 2> input(1,55);
  //    std::iota(input.begin(), input.end(),0);
  //    std::cout << input << '\n';
  //    source.push(input);
  //    source.push(input);
  //
  //    source.reset();
  //
  //
  //    FluidTensor<double, 2> ramp_to_99(1,100);
  //    std::iota(ramp_to_99.begin(),ramp_to_99.end(),0);
  //    source.push(ramp_to_99);
  //
  //    for(int i = 0 ; i < 10; ++i)
  //    {
  //        FluidTensor<double,2> output = source.pull(10);
  //        std::cout << output << '\n';
  //    }
  //
  //
  //
  //    source.reset();
  //    source.resize(1024);
  //
  //    FluidTensor<double, 2> cycle(1,64);
  //    std::iota(cycle.begin(), cycle.end(), 0);
  //    cycle.apply([](double&x){
  //        x =  std::cos(2 * M_PI * x * (1/64));
  //    });
  //
  //
  //    //Simulate data coming in smaller packets than being pulled out (implies
  //    a latency...)
  //    //should see in the console that available grows, and then gets consumed
  //    for(int i = 0; i < 64; ++i)
  //    {
  //        source.push(cycle);
  //        std::cout << "Before " << source.available() << '\n';
  //        while(source.available() >= 1024)
  //        {
  //            source.pull(1024);
  //        }
  //        std::cout << "After " << source.available() << '\n';
  //    }
  //
  ///************************************************************************
  // OLA end to end test
  // - Write in data to source, extract with hop
  // – window segment, ola into sink
  // – pull out from sink and compare with original
  // **************************************************************************/
  //
  //    size_t framesize = 1024;
  //    //read with overlap of four
  //    size_t hop = framesize >> 2;
  //    size_t n_frames = 64;
  //    size_t n_hops = ((framesize * n_frames) / hop);
  //
  //    size_t signal_length = framesize * n_frames;
  //    size_t padded_length = signal_length + framesize;
  //
  //    //Make a tensor for the signal, with half a frame's zero padding each
  //    end FluidTensor<double, 2> sine(1,padded_length);
  //    //Grab the bit without zero padding
  //    auto sig = sine(0,slice(framesize/2,signal_length));
  //    //Generate a cosine
  //    std::iota(sig.begin(), sig.end(), 0);
  //    sig.apply([](double& x){
  //        x =  std::cos(2 * M_PI * x * (1./64.));
  //    });
  //    //Make a window
  //    FluidTensor<double,2> window(1,framesize);
  //    std::vector<double>
  //    win=fluid::windows::windowFuncs[fluid::windows::WindowType::Hann](framesize);
  //    std::copy(win.begin(), win.end(), window.begin());
  //
  //    //Buffer in
  //    FluidSource<double> sine_in(padded_length);
  //    //Buffer out
  //    FluidSink<double> sine_out(1,padded_length);
  //    //Normalisation buffer for window
  //    FluidSink<double> norm_out(1,padded_length);
  //
  //    //push signal in
  //    sine_in.push(sine);
  //
  //    //Pull out frames from source, window and push in to sink
  //    for(int i = 0; i<n_hops;++i)
  //    {
  //        FluidTensor<double,2> frame = sine_in.pull(framesize,hop)
  //            .apply(window,[](double& x, double w){x*=w;});
  //
  //        norm_out.push(window,hop);
  //        sine_out.push(frame,hop);
  //    }
  //
  //    //Grab the whole result from sink
  //    FluidTensor<double,2> result = sine_out.pull(padded_length);
  //
  //    //Apply OLA normalisation
  //    result.apply(norm_out.pull(padded_length),[](double& x, double y)
  //                 {
  //                     if(x)
  //                         x /=  y > 0? y : 1 ;
  //                 });
  //
  //    //Subtract original from result
  //    result.apply(sine, [](double& x, double y){ x -= y; });
  //    //sum this difference
  //    double sum_of_diff = std::accumulate(result.begin(), result.end(), 0.0);
  //    //we want it to be small, kthx
  //    std::cout << "Difference summed " << sum_of_diff << '\n';
}
