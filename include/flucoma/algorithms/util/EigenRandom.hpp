#include "../../data/FluidIndex.hpp"
#include <Eigen/Core>
#include <cmath>
#include <optional>
#include <random>

namespace fluid::algorithm {

struct RandomSeed
{
  RandomSeed(index seed) : mSeed{seed} {}
  RandomSeed() : RandomSeed(-1) {}
  operator std::optional<size_t>()
  {
    return mSeed >= 0 ? mSeed : std::optional<index>{};
  }

  index mSeed;
};


template <typename T>
struct is_complex : std::false_type
{};
template <typename U>
struct is_complex<std::complex<U>> : std::true_type
{};
template <typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

template <typename T>
struct value_type_of
{
  using type = T;
};

template <typename T>
struct value_type_of<std::complex<T>>
{
  using type = T;
};

template <typename T>
using value_type_of_t = typename value_type_of<T>::type;

template <typename T>
struct Range
{
  T min;
  T max;
};

// deduction guide
template <typename T>
Range(T, T) -> Range<T>;

template <typename T>
Range<T> DefaultRange =
    Range<T>{std::numeric_limits<T>::min(), std::numeric_limits<T>::max()};

template <>
auto DefaultRange<double> = Range<double>{-1.0, 1.0};

template <>
auto DefaultRange<float> = Range<double>{-1.0f, 1.0f};


// Wraps up C++ random stuff to suit our needs
// can cope with uniform distributions of integral or real or complex real types
// replciates Eigen::Random default ranges (-1:1 for reals, -max:max for
// integral types) but also allows settable ranges (unlike Eigen::Random)
template <typename T,
          typename D = std::conditional_t<
              std::is_integral_v<T>, std::uniform_int_distribution<T>,
              std::uniform_real_distribution<value_type_of_t<T>>>>
struct RandomGenerator
{
  RandomGenerator(std::optional<size_t> seed, Range<value_type_of_t<T>> range)
      : g{seed ? *seed : rd()}, d{range.min, range.max}
  {}
  RandomGenerator(std::optional<size_t> seed)
      : RandomGenerator(seed, DefaultRange<value_type_of_t<T>>)
  {}
  RandomGenerator(const RandomGenerator& x) : g{x.g}, d{x.d} {}

  T operator()() const
  {
    // if complex, we need to generate two random values
    if constexpr (!is_complex_v<T>)
      return d(g);
    else
      return T{d(g), d(g)};
  }

private:
  std::random_device      rd;
  mutable std::mt19937_64 g;

  mutable D d;
};


template <typename EigenType>
auto EigenRandom(index rows, index cols, RandomSeed seed,
                 Range<value_type_of_t<typename EigenType::Scalar>> range)
{
  return EigenType::NullaryExpr(
      rows, cols, RandomGenerator<typename EigenType::Scalar>{seed, range});
}

template <typename EigenType>
auto EigenRandom(index rows, index cols, RandomSeed seed)
{
  using T = value_type_of_t<typename EigenType::Scalar>;
  return EigenRandom<EigenType>(rows, cols, seed, DefaultRange<T>);
}

template <typename EigenType>
auto EigenRandom(index size, RandomSeed seed,
                 Range<value_type_of_t<typename EigenType::Scalar>> range)
{
  static_assert(EigenType::RowsAtCompileTime == 1 ||
                    EigenType::ColsAtCompileTime == 1,
                "Eigen Type is not a vector, did you mean to call (rows,cols) "
                "version?");
  if constexpr (EigenType::ColsAtCompileTime == 1)
  {
    return EigenType::NullaryExpr(
        size, 1, RandomGenerator<typename EigenType::Scalar>{seed, range});
  }
  if constexpr (EigenType::RowsAtCompileTime == 1)
  {
    return EigenType::NullaryExpr(
        1, size, RandomGenerator<typename EigenType::Scalar>{seed, range});
  }
}

template <typename EigenType>
auto EigenRandom(index size, RandomSeed seed)
{
  using T = value_type_of_t<typename EigenType::Scalar>;
  return EigenRandom<EigenType>(size, seed, DefaultRange<T>);
}

template <typename EigenType>
auto EigenRandomPhase(index rows, index cols, RandomSeed seed)
{

  static_assert(is_complex_v<typename EigenType::Scalar>,
                "This only works for complex valued Eigen types");

  constexpr double twopi = 2 * M_PI;
  using T = value_type_of_t<typename EigenType::Scalar>;
  auto f = [rg = RandomGenerator<T>{seed, Range{0.0, twopi}}]() {
    return std::polar(1.0, rg());
  };

  return EigenType::NullaryExpr(rows, cols, f);
}


} // namespace fluid::algorithm