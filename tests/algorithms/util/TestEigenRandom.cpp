#define CATCH_CONFIG_MAIN
#include <Eigen/Core>
#include <catch2/catch_all.hpp>
#include <flucoma/algorithms/util/EigenRandom.hpp>
#include <complex>
#include <iostream>
#include <optional>
#include <random>

namespace fluid::algorithm {

TEST_CASE("EigenRandom Can Make Basic Random Containers")
{

  SECTION("Matrix of Double")
  {
    Eigen::MatrixXd mrnd = EigenRandom<Eigen::MatrixXd>(3, 4, RandomSeed{});

    // Right size
    REQUIRE((mrnd.rows() == 3 && mrnd.cols() == 4));
    // No NaNs or infs
    REQUIRE((mrnd.array().isFinite()).all());
    // Not constant value
    auto first = mrnd(0, 0);
    REQUIRE_FALSE((mrnd.array() == first).all());
  }

  SECTION("2D Array of Double")
  {
    Eigen::ArrayXXd mrnd = EigenRandom<Eigen::ArrayXXd>(3, 4, RandomSeed{});

    // Right size
    REQUIRE((mrnd.rows() == 3 && mrnd.cols() == 4));
    // No NaNs or infs
    REQUIRE((mrnd.array().isFinite()).all());
    // Not constant value
    auto first = mrnd(0, 0);
    REQUIRE_FALSE((mrnd.array() == first).all());
  }

  SECTION("Vector of Double")
  {
    Eigen::VectorXd vrnd = EigenRandom<Eigen::VectorXd>(4, RandomSeed{});

    // Right size
    REQUIRE((vrnd.size() == 4));
    // No NaNs or infs
    REQUIRE((vrnd.array().isFinite()).all());
    // Not constant value
    auto first = vrnd(0);
    REQUIRE_FALSE((vrnd.array() == first).all());
  }

  SECTION("Vector of int")
  {
    Eigen::VectorXi vrnd = EigenRandom<Eigen::VectorXi>(4, RandomSeed{});

    // Right size
    REQUIRE((vrnd.size() == 4));
    // Not constant value
    auto first = vrnd(0);
    REQUIRE_FALSE((vrnd.array() == first).all());
  }

  SECTION("1D Array of Double")
  {
    Eigen::ArrayXd mrnd = EigenRandom<Eigen::ArrayXd>(4, RandomSeed{});

    // Right size
    REQUIRE(mrnd.size() == 4);
    // No NaNs or infs
    REQUIRE((mrnd.array().isFinite()).all());
    // Not constant value
    auto first = mrnd(0);
    REQUIRE_FALSE((mrnd.array() == first).all());
  }

  SECTION("2D Array of Complex Double")
  {
    Eigen::ArrayXXcd mrnd = EigenRandom<Eigen::ArrayXXcd>(3, 4, RandomSeed{});

    // Right size
    REQUIRE((mrnd.rows() == 3 && mrnd.cols() == 4));
    // No NaNs or infs
    REQUIRE((mrnd.array().isFinite()).all());
    // Not constant value
    auto first = mrnd(0, 0);
    REQUIRE_FALSE((mrnd.array() == first).all());
  }
}

TEST_CASE("EigenRandom Can Manually Seet Random Seed For Repeatability")
{

  Eigen::MatrixXd mrnd1 = EigenRandom<Eigen::MatrixXd>(3, 4, RandomSeed{42});
  Eigen::MatrixXd mrnd2 = EigenRandom<Eigen::MatrixXd>(3, 4, RandomSeed{42});

  // Same seed -> same result
  REQUIRE((mrnd1.array() == mrnd2.array()).all());

  // Different seed -> different result
  Eigen::MatrixXd mrnd3 = EigenRandom<Eigen::MatrixXd>(3, 4, RandomSeed{4203});
  REQUIRE_FALSE((mrnd1.array() == mrnd3.array()).all());
}

TEST_CASE("Eigen Random Can Set Min and Max")
{
  double          min = -0.4;
  double          max = 0.1;
  Eigen::MatrixXd mrnd =
      EigenRandom<Eigen::MatrixXd>(3, 4, RandomSeed{42}, Range{-0.4, 0.1});

  REQUIRE((mrnd.minCoeff() >= min && mrnd.maxCoeff() < max));
}

TEST_CASE("EigenRandom Can make complex array of random phase, unit magnitude")
{
  Eigen::ArrayXXcd a = EigenRandomPhase<Eigen::ArrayXXcd>(4, 3, RandomSeed{});
  constexpr double twopi = 2 * M_PI;

  std::cout << a.arg();
  REQUIRE((a.arg().maxCoeff() - a.arg().minCoeff() <= twopi));
  REQUIRE_THAT(a.abs().reshaped(),
               Catch::Matchers::AllMatch(Catch::Matchers::WithinRel(1.0)));
}

} // namespace fluid::algorithm
