/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

/*
This program demonstrates the use of the fluid decomposition toolbox
to apply an algorithm on an input dataset
*/

#include <flucoma/algorithms/public/UMAP.hpp>
#include <flucoma/data/FluidDataSet.hpp>
#include <flucoma/data/FluidIndex.hpp>
#include <flucoma/data/FluidJSON.hpp>
#include <flucoma/data/FluidMemory.hpp>
#include <flucoma/data/TensorTypes.hpp>

#include <Eigen/Core>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
  using namespace fluid;
  using namespace fluid::algorithm;
  using fluid::index;

  if (argc != 3)
  {
    std::cerr << "usage: umap input_file.json output_file.json\n";
    return 1;
  }

  FluidDataSet<std::string, double, 1> datasetIN(1);
  FluidDataSet<std::string, double, 1> datasetOUT(1);

  const char* inputFile = argv[1];
  const char* outputFile = argv[2];

  auto           inputJSON = JSONFile(inputFile, "r");
  nlohmann::json j = inputJSON.read();

  if (!inputJSON.ok())
  {
    std::cerr << "failed to read input " << inputFile << "\n";
    return 2;
  }

  if (!check_json(j, datasetIN))
  {
    std::cerr << "Invalid JSON format\n";
    return 3;
  }

  datasetIN = j.get<FluidDataSet<std::string, double, 1>>();

  algorithm::UMAP algorithm;

  datasetOUT = algorithm.train(datasetIN, 15, 2, 0.1, 200, 0.1);

  auto outputJSON = JSONFile(outputFile, "w");
  outputJSON.write(datasetOUT);

  if (!outputJSON.ok())
  {
    std::cerr << "failed to write output to " << outputFile << "\n";
  }

  return 0;
}
