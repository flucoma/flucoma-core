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

#include "../util/Assign2D.hpp"
#include "../util/DistanceFuncs.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/Munkres.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class Grid
{
public:
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;
  using DataSet = FluidDataSet<std::string, double, 1>;

  DataSet process(DataSet& in, index overSample = 1, index extent = 0,
                  index axis = 0)
  {
    using namespace Eigen;
    using namespace _impl;
    using namespace std;

    assert(in.dims() == 2);
    index    N = in.size();
    index    M = N * overSample;
    ArrayXXd data = asEigen<Array>(in.getData());
    double   xMin = data.col(0).minCoeff();
    double   xMax = data.col(0).maxCoeff();
    double   yMin = data.col(1).minCoeff();
    double   yMax = data.col(1).maxCoeff();
    double   area = (xMax - xMin) * (yMax - yMin);
    double   size = static_cast<double>(N);
    double   step = sqrt(area / M);
    index    numCols, numRows;

    if (area <= 0) return DataSet();

    if (extent > 0)
    {
      numCols = (axis == 0) ? extent : lrint(ceil(size / extent));
      numRows = (axis == 1) ? extent : lrint(ceil(size / extent));
    }
    else
    {
      numCols = lrint(ceil((xMax - xMin) / step));
      numRows = lrint(ceil((yMax - yMin) / step));
    }

    ArrayXd colPos, rowPos;
    if (extent > 0 && axis == 1)
    {
      rowPos = ArrayXidx::LinSpaced(M, 0, M - 1)
                   .unaryExpr([&](const index x) { return x % numRows; })
                   .cast<double>();
      colPos = (ArrayXidx::LinSpaced(M, 0, M - 1) / numRows).cast<double>();
    }
    else
    {
      colPos = ArrayXidx::LinSpaced(M, 0, M - 1)
                   .unaryExpr([&](const index x) { return x % numCols; })
                   .cast<double>();
      rowPos = (ArrayXidx::LinSpaced(M, 0, M - 1) / numCols).cast<double>();
    }

    ArrayXd  xPos = xMin + (colPos / (numCols - 1)) * (xMax - xMin);
    ArrayXd  yPos = yMin + (rowPos / (numRows - 1)) * (yMax - yMin);
    ArrayXXd grid(M, 2);
    grid << xPos, yPos;
    ArrayXXd cost = algorithm::DistanceMatrix<ArrayXXd>(data, grid, 1);
    ArrayXidx  assignment(N);
    bool     outcome = assign2D.process(cost, assignment);
    if (!outcome) return DataSet();

    DataSet    result(2);
    auto       ids = in.getIds();
    RealVector asignedPos(2);
    for (index i = 0; i < N; i++)
    {
      asignedPos(0) = colPos(assignment(i));
      asignedPos(1) = rowPos(assignment(i));
      auto id = ids(i);
      result.add(ids(i), asignedPos);
    }
    return result;
  }

private:
  Assign2D assign2D;
};
}// namespace algorithm
}// namespace fluid
