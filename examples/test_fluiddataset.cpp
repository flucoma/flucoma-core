#include <fstream>

#include <data/FluidTensor.hpp>
#include <data/FluidDataset.hpp>
#include <data/TensorTypes.hpp>


using fluid::FluidTensor;
using fluid::FluidTensorView;
using std::string;

int main(int argc, char *argv[])
{
  using std::cout;
  using std::vector;

  //FluidTensor<double, 1> p1{{1., 2., 34., 5., 6., 7.}};
  FluidTensor<double, 2> p1{{1., 2.}, {34., 5.}, {6., 7.}};
  cout<<p1.descriptor().extents[0]<<std::endl;
  fluid::FluidDataset<double, std::string, 2> ds(3, 2);
  bool result = ds.add("my first point", p1);
  std::cout<<"add result "<<result<<std::endl;

  FluidTensor<double, 2> p2(3, 2);
  result = ds.get("my first point", p2);
  std::cout<<"get result "<<result<<std::endl;
  std::cout<<p2<<std::endl;

  FluidTensor<double, 2> p3{{3.,3.},{3.,3.},{3.,3.}};
  result = ds.update("my first point", p3);
  std::cout<<"update result "<<result<<std::endl;
  ds.get("my first point", p2);
  std::cout<<p2<<std::endl;

  result = ds.remove("my first point");
  std::cout<<"remove result "<<result<<std::endl;
  result = ds.get("my first point", p2);
  std::cout<<"get result "<<result<<std::endl;
  return 0;
}
