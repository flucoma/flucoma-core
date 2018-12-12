#include <Eigen/Dense>
#include <algorithms/util/MedianFilter.hpp>

int main(int argc, char *argv[]) {
  using Eigen::ArrayXd;
  using fluid::algorithm::MedianFilter;
  using std::cout;
  using std::endl;
  using std::vector;


  ArrayXd in(15);
  in << 1.0, 0.5, 0.2, 10.4, 6.5, 7.0, 9.5, 5.5, 3.5, 6.2, 0.1, 0.4, 0.01, 2.0,
      3.0;
  ArrayXd out(15);
  MedianFilter mf(in, 3);

  mf.process(out);

  for (int i = 0; i < in.size(); i++) {
    cout << "in " << in[i] << " out " << out[i] << std::endl;
  }
  cout<<"============"<<std::endl;

  mf.insertRight(0.9);
  mf.insertRight(0.5);
  mf.insertRight(0.3);
  mf.process(out);
  for (int i = 0; i < in.size(); i++) {
    cout << " out " << out[i] << std::endl;
  }
  cout<<"============"<<std::endl;

  return 0;
}
