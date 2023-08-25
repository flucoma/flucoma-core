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

template <typename dataType>
class DTW
{
    using Matrix =   Eigen::Matrix<dataType, -1, -1>;
    using Vector =   Eigen::Matrix<dataType, -1, 1>;
    using Array =    Eigen::Array<dataType, -1, 1>;
    using PathType = Eigen::Matrix<index, 2, -1>;

public:
    explicit DTW() = default;
    ~DTW() = default;

    static dataType process(FluidTensorView<const dataType, 2> x1, 
                            FluidTensorView<const dataType, 2> x2,
                            index p = 2)
    {
        Matrix distanceMetrics(x1.rows(), x2.rows());

        return calculateDistanceMetrics(
            _impl::asEigen<Array>(x1), 
            _impl::asEigen<Array>(x2)
            distanceMetrics,
            p
        );
    }
    
private:
    static dataType calculateDistanceMetrics(Eigen::Ref<Matrix> x1, 
                                             Eigen::Ref<Matrix> x2,
                                             Eigen::Ref<Matrix> distance, index p)
    {
        distance.conservativeResize(x1.rows(), x2.rows());
        // simple brute force DTW is very inefficient, see FastDTW
        for (index i = 0; i < x1.rows(); i++)
        {
            for (index j = 0; j < x2.rows(); j++)
            {
                Array x1i = x1.row(i);
                Array x2j = x2.row(j);

                distance(i, j) = differencePNormToTheP(x1i, x2j, p);

                if (i > 0 || j > 0)
                {
                    dataType minimum = std::numeric_limits<dataType>::max();

                    if (i > 0 && j > 0)
                        minimum = std::min(minimum, distance(i-1, j-1));
                    if (i > 0)
                        minimum = std::min(minimum, distance(i-1, j  ));
                    if (j > 0)
                        minimum = std::min(minimum, distance(i  , j-1));

                    distance(i, j) += minimum;
                }
            }
        }

        return std::pow(distance(x1.rows() - 1, x2.rows() - 1), 1.0 / p);
    }

    // P-Norm of the difference vector
    // Lp{vec} = (|vec[0]|^p + |vec[1]|^p + ... + |vec[n-1]|^p + |vec[n]|^p)^(1/p)
    // i.e., the 2-norm of a vector is the euclidian distance from the origin
    //       the 1-norm is the sum of the absolute value of the elements
    // To the power P since we'll be summing multiple Norms together and they
    // can combine into a single norm if you calculate the norm of multiple norms (normception)
    inline static dataType differencePNormToTheP(const Eigen::Ref<const Vector>& v1, 
                                                 const Eigen::Ref<const Vector>& v2, index p)
    {
        // assert(v1.size() == v2.size());
        return (v1.array() - v2.array()).abs().pow(p).sum();
    }
};

} // namespace algorithm
} // namespace fluid