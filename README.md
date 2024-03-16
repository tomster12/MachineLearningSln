# Machine Learning

Machine learning library and implementation in C++ in VS 22.

TBMLGeneticAlgorithm and TBMLMNISTDrawer both use 32-bit SFML, see below for setup.

TBMLNeuralNetwork MNIST examples only work in 32-bit.

## Setup

1. Download 32-bit SFML from [the website](https://www.sfml-dev.org/).

2. Place **/include** and **/lib** libraries here:

   - `dependencies/SFML/include/SFML`
   - `dependencies/SFML/lib`

3. Place **/bin/X-2.dll** here:

   - `bin/TBMLMNISTDrawer/output/Release`
   - `bin/TBMLGeneticAlgorithm/output/Release`


3. Place **/bin/X-d-2.dll** here:

   - `bin/TBMLMNISTDrawer/output/Debug`
   - `bin/TBMLGeneticAlgorithm/output/Debug`
