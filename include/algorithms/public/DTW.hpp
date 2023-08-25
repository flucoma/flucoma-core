/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidDataSet.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include "../../data/FluidMemory.hpp"
#include <Eigen/Core>
#include <random>

namespace fluid {
namespace algorithm {


// debt of gratitude to the wonderful article on https://rtavenar.github.io/blog/dtw.html
// a better explanation of DTW than any other algorithm explanation I've seen

class DTW
{
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    using ArrayXd =  Eigen::ArrayXd;

public:
    explicit DTW() = default;
    ~DTW() = default;

    // functions so the DataClient<DTW> doesnt have freak out
    void init()  const {}
    void clear() const {}

    constexpr index size()        const { return 0; }
    constexpr index dims()        const { return 0; }
    constexpr index initialized() const { return true; }

    double process(InputRealMatrixView x1, InputRealMatrixView x2, index p = 2) const
    {
        distanceMetrics.conservativeResize(x1.rows(), x2.rows());
        // simple brute force DTW is very inefficient, see FastDTW
        for (index i = 0; i < x1.rows(); i++)
        {
            for (index j = 0; j < x2.rows(); j++)
            {
                ArrayXd x1i = _impl::asEigen<Eigen::Array>(x1.row(i));
                ArrayXd x2j = _impl::asEigen<Eigen::Array>(x2.row(j));

                distanceMetrics(i, j) = differencePNormToTheP(x1i, x2j, p);

                if (i > 0 || j > 0)
                {
                    double minimum = std::numeric_limits<double>::max();

                    if (i > 0 && j > 0) 
                        minimum = std::min(minimum, distanceMetrics(i - 1, j - 1));
                    if (i > 0)
                        minimum = std::min(minimum, distanceMetrics(i - 1, j));
                    if (j > 0)
                        minimum = std::min(minimum, distanceMetrics(i, j - 1));

                    distanceMetrics(i, j) += minimum;
                }
            }
        }

        return std::pow(distanceMetrics.bottomLeftCorner<1, 1>().value(), 1.0 / p);
    }

private:
    mutable MatrixXd distanceMetrics;

    // P-Norm of the difference vector
    // Lp{vec} = (|vec[0]|^p + |vec[1]|^p + ... + |vec[n-1]|^p + |vec[n]|^p)^(1/p)
    // i.e., the 2-norm of a vector is the euclidian distance from the origin
    //       the 1-norm is the sum of the absolute value of the elements
    // To the power P since we'll be summing multiple Norms together and they
    // can combine into a single norm if you calculate the norm of multiple norms (normception)
    inline static double differencePNormToTheP(const Eigen::Ref<const VectorXd>& v1, const Eigen::Ref<const VectorXd>& v2, index p)
    {
        // assert(v1.size() == v2.size());
        return (v1 - v2).array().abs().pow(p).sum();
    }
};

} // namespace algorithm
} // namespace fluid