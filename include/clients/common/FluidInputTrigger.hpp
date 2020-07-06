#pragma once 

namespace fluid {
namespace client {
    
class FluidInputTrigger
{
public:
  template<typename T, typename F> 
  void process(std::vector<FluidTensorView<T, 1>> &input,
               std::vector<FluidTensorView<T, 1>> &output, 
               F&& func) 
  {  
    if(!input[0].data() || !output[0].data()) return;
    
    double trig = input[0][0]; 
    
    if(trig > 0 && !(mPreviousTrigger > 0))
    {
      func();
      output[0](0)  = 1;
      
    } else output[0](0)  = 0;
    
    
    mPreviousTrigger = trig; 
  }
  
  void reset() { mPreviousTrigger = 0; }
private:
  double mPreviousTrigger{0};
}; 
}  
}
