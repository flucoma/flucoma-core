#define CATCH_CONFIG_MAIN
#include <flucoma/algorithms/public/MLP.hpp>
#include <flucoma/algorithms/public/SGD.hpp>
#include <flucoma/algorithms/util/NNFuncs.hpp>
#include <catch2/catch_all.hpp>
#include <flucoma/data/FluidJSON.hpp> 
#include <flucoma/clients/nrt/MLPClassifierClient.hpp>
#include <flucoma/data/FluidIndex.hpp>
#include <flucoma/data/FluidJSON.hpp>
#include <flucoma/data/FluidTensor.hpp>
#include <flucoma/data/FluidDataSetSampler.hpp>
#include <iostream>

namespace fluid::algorithm {

using Array = Eigen::ArrayXXd;
using Vector = Eigen::ArrayXd;

auto manual(double alpha = 0.0,
            double eta = 0.1) -> std::tuple<Array, Array, Vector, Vector>
{

  Eigen::ArrayXd x(3);
  x << 0.6, 0.8, 0.7;

  Eigen::ArrayXXd W(3, 2);
  W << 0.1, 0.2, 0.3, 0.1, 0.5, 0.0;

  Eigen::ArrayXXd W2(2, 1);
  W2 << 0.1, 0.2;

  Eigen::ArrayXd b1(2);
  b1 << 0.1, 0.1;

  Eigen::ArrayXd b2(1);
  b2 << 1.0;


  double h1 //= g(X1 * W_i1 + b11) = g(0.6 * 0.1 + 0.8 * 0.3 + 0.7 * 0.5 + 0.1)
      = 0.679178699175393;
  double h2 //= g(X2 * W_i2 + b12) = g(0.6 * 0.2 + 0.8 * 0.1 + 0.7 * 0 + 0.1)
      = 0.574442516811659;
  double o1 //= g(h * W2 + b21) = g(0.679 * 0.1 + 0.574 * 0.2 + 1)
      = 0.7654329236196236;
  double d21 = (1 - 0.765) * 0.765 * (2 * 0.765);  // = 0.17953522             
  double d11 = (1 - 0.679) * 0.679 * d21 * 0.1; //= 0.00391198 
  double d12 = (1 - 0.574) * 0.574 * d21 * 0.2; // =0.00879186042
  
  // keeping the alphas here, even though we don't yet have regularization 
  double
      W1grad11 = // X1 * d11 + alpha * W11 = 0.6 * 0.00391198  + 0. * 0.1 = 0.0023510147535
      0.6 * d11 + alpha * 0.1;
  double
      W1grad12 //= X1 * d12 + alpha * W12 = 0.6 * 0.00879186042 + 0. * 0.2 = 0.00526667
      = 0.6 * d12 + alpha * 0.2;
  double
      W1grad21 //= X2 * d11 + alpha * W13 = 0.8 * 0.00391198 + 0. * 0.3 = 0.043336
      = 0.8 * d11 + alpha * 0.3;
  double
      W1grad22 //= X2 * d12 + alpha * W14 = 0.8 * 0.00879186042 + 0. * 0.1 = 0.03992
      = 0.8 * d12 + alpha * 0.1;
  double
      W1grad31 //= X3 * d11 + alpha * W15 = 0.6 * 0.00391198  + 0. * 0.5 = 0.060002
      = 0.7 * d11 + alpha * 0.5;
  double W1grad32 //= X3 * d12 + alpha * W16 = 0.6 * 0.00879186042 + 0. * 0 = 0.02244
      = 0.7 * d12 + alpha * 0;

  double W2grad1 //= h1 * d21 + alpha * W21 = 0.679 * 0.17953522 + 0. * 0.1 = 0.5294
      = h1 * d21 + alpha * 0.1;
  double
      W2grad2 //= h2 * d21 + alpha * W22 = 0.574 * 0.17953522 + 0.1 * 0.2 = 0.45911
      = h2 * d21 + alpha * 0.2;
  double b1grad1 = d11; // = 0.00391198 
  double b1grad2 = d12; // = 0.00879186042
  double b2grad = d21;  // = 0.17953522   

  Eigen::ArrayXXd W1grad(3, 2);
  W1grad << W1grad11, W1grad12, W1grad21, W1grad22, W1grad31, W1grad32;

  Eigen::ArrayXXd W1_t1 = W - eta * W1grad;

  Eigen::ArrayXXd W2grad(2, 1);
  W2grad << W2grad1, W2grad2;

  Eigen::ArrayXXd W2_t1 = W2 - eta * W2grad;

  Eigen::ArrayXd b1grad(2);
  b1grad << b1grad1, b1grad2;

  Eigen::ArrayXd b1_t1 = b1 - eta * b1grad;

  Eigen::ArrayXd b2_t1 = b2 - eta * b2grad;
  return {W1_t1, W2_t1, b1_t1, b2_t1};
}

TEST_CASE("MLP works on precomputed example")
{
  // Based on premise from 
  // https://github.com/scikit-learn/scikit-learn/blob/d666202a9349893c1bd106cc9ee0ff0a807c7cf3/sklearn/neural_network/tests/test_mlp.py
  // although I take Torch as the gospel here 

  // Our 'data' 
  FluidTensor<double, 2> x = {{0.6, 0.8, 0.7}};
  FluidTensor<double, 2> y = {{0}};

  // Make a network and set initial conditions
  MLP   mlp = MLP();
  index act = static_cast<index>(NNActivations::Activation::kSigmoid);
  mlp.init(3, 1, {2}, act, act);
  FluidTensor<double, 2> layer0Coeffs = {{0.1, 0.2}, {0.3, 0.1}, {0.5, 0}};
  FluidTensor<double, 2> layer1Coeffs = {{0.1}, {0.2}};
  FluidTensor<double, 1> layer0Bias = {0.1, 0.1};
  FluidTensor<double, 1> layer1Bias = {1.0};

  mlp.setParameters(0, layer0Coeffs, layer0Bias, act);
  mlp.setParameters(1, layer1Coeffs, layer1Bias, act);

  // train for a single iteration 
  SGD sgd;
  sgd.train(mlp, x, y, 1, 1, 0.1, 0.0, 0.0);
  
  // get our hand computed data 
  auto [W1, W2, b1, b2] = manual(0.0, 0.1);

  // compare ours with hand computed 
  using Catch::Matchers::WithinAbs;
  FluidTensor<double, 2> W1_learned(3, 2);
  FluidTensor<double, 2> W2_learned(2, 1);
  FluidTensor<double, 1> b1_learned(2);
  FluidTensor<double, 1> b2_learned(1);
  index                  dummy{0};
  mlp.getParameters(0, W1_learned, b1_learned, dummy);
  mlp.getParameters(1, W2_learned, b2_learned, dummy);

  for (index i = 0; i < 3; ++i)
    for (index j = 0; j < 2; ++j)
    {
      double arg = W1_learned(i, j);
      double target = W1(i, j);
      auto   matcher = WithinAbs(target, 1e-3);
      CHECK_THAT(arg, matcher);
    }

  for (index i = 0; i < 2; ++i)
    for (index j = 0; j < 1; ++j)
    {
      double arg = W2_learned(i, j);
      double target = W2(i, j);
      auto   matcher = WithinAbs(target, 1e-3);
      CHECK_THAT(arg, matcher);
    }

  for (index i = 0; i < 2; ++i)
  {
      double arg = b1_learned(i);
      double target = b1(i);
      auto   matcher = WithinAbs(target, 1e-3);
      CHECK_THAT(arg, matcher);
    }

  for (index i = 0; i < 1; ++i)
  {
      double arg = b2_learned(i);
      double target = b2(i);
      auto   matcher = WithinAbs(target, 1e-3);
      CHECK_THAT(arg, matcher);
    }

  FluidTensor<double, 1> yy{0};
  mlp.processFrame(x.row(0), yy, 0, 2);
  REQUIRE_THAT(yy[0], WithinAbs(0.7565, 1e-3));
}

std::pair<FluidDataSet<std::string, index, 1>,FluidDataSet<std::string, index, 1>> makeUnalignedDataSets(index size){
  
  FluidTensor<std::string, 1> ids_input(size);
  FluidTensor<index, 2> data_input(size,1);

  std::generate(ids_input.begin(), ids_input.end(),
                [n = 0]() mutable { return std::to_string(n++); });
  
  std::generate(data_input.begin(), data_input.end(), [n = 0]() mutable { return n++; });
  
  FluidDataSet<std::string, index,1> dataset_in (ids_input, data_input); 

  //shuffle outputs 
  std::vector<index> lookup(size); 
  std::iota(lookup.begin(), lookup.end(), 0); 
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(lookup.begin(), lookup.end(), g); 

  FluidTensor<std::string, 1> ids_output(size);
  FluidTensor<index, 2> data_output(size,1);

  std::transform(lookup.begin(), lookup.end(), ids_output.begin(),
                 [&lookup, &ids_input](index i) { return ids_input[i]; });

  std::transform(lookup.begin(), lookup.end(), data_output.begin(),
                 [&lookup, &data_input](index i) { return data_input(i,0); });
  
  FluidDataSet<std::string, index,1> dataset_out (ids_output, data_output); 

  return { dataset_in, dataset_out }; 

}


TEST_CASE("Test batch loader for mismatched fluid datasets")
{  
  const index         batchSize = 64;
  const index         N = 300;
  auto                data = makeUnalignedDataSets(N);
  index               datacount = 0;
  FluidDataSetSampler ds(data.first, data.second, 64, 0, true);
  
  REQUIRE_FALSE(ds.begin() == ds.end());

  auto  inputs = data.first.getData().col(0);
  auto  outputs = data.second.getData().col(0);
  index i = 0;
  for (auto batch : ds)
  {
    std::cout << "ping\n";
    index expectedSize = i++ == 0 ? batchSize + (N % batchSize) : batchSize;
    CHECK(batch->rows() == expectedSize);
    auto inputidx = batch->col(0);
    auto outputidx = batch->col(1);
    for (index j = 0; j < batch->rows(); ++j)
    {
      CHECK(inputs[inputidx[j]] == outputs[outputidx[j]]);
    }
  }
}


} // namespace fluid::algorithm