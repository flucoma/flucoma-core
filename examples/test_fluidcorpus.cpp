#include <fstream>

#include <clients/FluidCorpusClient.hpp>
#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>
#include <clients/common/Result.hpp>

using fluid::FluidTensor;
using fluid::FluidTensorView;
using std::string;

int main(int argc, char *argv[])
{
  using std::cout;
  using std::vector;
  using fluid::client::Result;


  fluid::client::FluidCorpusClient client(6);
  std::cout<<"adding points"<<std::endl;
  FluidTensor<double, 1> point{{1., 2., 34., 5., 6., 7.}};
  Result r1 =  client.addPoint("p1", point);
  Result r2 =  client.addPoint("p2", point);
  std::cout<<"result "<<r1.message()<<std::endl;
  std::cout<<"result "<<r2.message()<<std::endl;

  std::cout<<"getting invalid point"<<std::endl;
  FluidTensor<double, 1> newPoint = FluidTensor<double, 1>(6);
  Result r3 =  client.getPoint("p233", newPoint);
  std::cout<<"result "<<r3.message()<<std::endl;

  std::cout<<"updating point "<<std::endl;
  FluidTensor<double, 1> point2{{3., 3., 3., 3., 3., 3.}};
  Result r4 =  client.updatePoint("p2", point2);
  std::cout<<"result "<<r4.message()<<std::endl;
  Result r5 =  client.getPoint("p2", newPoint);
  std::cout<<newPoint<<std::endl;
  return 0;
}
