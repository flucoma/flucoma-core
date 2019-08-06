#include <fstream>

#include <algorithms/KDTree.hpp>
#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>


using fluid::FluidTensor;
using fluid::FluidTensorView;
using std::string;

int main(int argc, char *argv[])
{
  using std::cout;
  using std::vector;


  fluid::algorithm::KDTree tree(2);

  FluidTensor<double, 1> p1{{1., 2.}};
  FluidTensor<double, 1> p2{{4., 3.}};
  FluidTensor<double, 1> p3{{7.5, 1.5}};
  FluidTensor<double, 1> p4{{0.5, 8.}};
  FluidTensor<double, 1> p5{{0.2, 7.}};
  FluidTensor<double, 1> p6{{0.7, 5.}};
  tree.addNode("p1",p1);
  tree.addNode("p2",p2);
  tree.addNode("p3",p3);
  tree.addNode("p4",p4);
  tree.addNode("p5",p5);
  tree.addNode("p6",p6);
  FluidTensor<double, 1> test{{9, 5.}};
  string n = tree.nearest(test);
  std::cout<<n<<std::endl;
  return 0;
}
