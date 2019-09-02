#include <fstream>

#include <algorithms/KDTree.hpp>
#include <data/FluidTensor.hpp>
#include <data/FluidDataset.hpp>
#include <data/TensorTypes.hpp>
#include <random>


int main(int argc, char *argv[])
{
  using fluid::FluidTensor;
  using fluid::FluidTensorView;
  using fluid::FluidDataset;
  using std::string;
  using std::cout;
  using std::vector;


  fluid::algorithm::KDTree<string> tree(2);

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
  //FluidTensor<string, 1> result = tree.kNearest(test, 3);
  auto result = tree.kNearest(test, 3);
  std::cout<<result.getData()(0, 0)<<std::endl;
  std::cout<<result.getData()(1, 0)<<std::endl;
  std::cout<<result.getData()(2, 0)<<std::endl;
  std::cout<<"-----"<<std::endl;
  tree.print();
  int dim = 13;
  FluidDataset<std::string, double, std::string, 1> ds(dim);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  for (int i = 0; i < 10000; i++) {
    FluidTensor<double, 1> p(dim);
    for (int d = 0; d < dim; d++) {
      p(d) =  dis(gen);
    }
    //std::cout<<"p"+std::to_string(i)<<" "<<p<<std::endl;
    ds.add("p"+std::to_string(i), p, "p"+std::to_string(i));
  }

  fluid::algorithm::KDTree<string> tree2(ds);
  std::cout<<"-----"<<std::endl;
  tree2.print();
  FluidTensor<double, 1> test1(dim);
  for (int d = 0; d < dim; d++) test1(d) = 0;
  FluidTensor<double, 1> r(dim);
  auto result1 = tree2.kNearest(test1, 3);
  for(int i = 0; i<3;i++){
    ds.get(result1.getTargets()(i), r);
    std::cout<<r<<std::endl;
  }
  //std::cout<<ds.get(result1(1))<<std::endl;
  //std::cout<<ds.get(result1(2))<<std::endl;


  return 0;
}
