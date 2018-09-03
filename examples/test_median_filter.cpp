#include "algorithms/MedianFilter.hpp"
#include <Eigen/Dense>

int main(int argc, char *argv[]) {
  using Eigen::ArrayXd;
  using fluid::medianfilter::MedianFilter;
  using std::cout;
  using std::endl;
  using std::vector;

  MedianFilter mf(3);
  ArrayXd in(15);
  in << 1.0, 0.5, 0.2, 10.4, 6.5, 7.0, 9.5, 5.5, 3.5, 6.2, 0.1, 0.4, 0.01, 2.0,
      3.0;
  ArrayXd out(15);
  mf.process(in, out);
  for (int i = 0; i < in.size(); i++) {
    cout << "in " << in[i] << " out " << out[i] << std::endl;
  }
  return 0;
}
