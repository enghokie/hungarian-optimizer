# Hungarian Optimizer

> A c++ implementation of the Hungarian algorithm.

## Dependency

- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) (V3.3.7 is used here, other versions may also work.)
- [DocTest](https://github.com/doctest/doctest) Just the header file is needed:
  `wget https://github.com/doctest/doctest/releases/download/v2.4.9/doctest.h`

## Usage

Just run the `run.sh` script under the directory of repository.

```bash
./run.sh
```

This will execute the DocTest tests from the `TestHungarian.cpp`

### Inputs/Outputs

The `HungarianOptimizer` class stores all the essential data to compute the
optimal job assignments from a given cost matrix, therefore an object of this
class is the entry point to all the functionality. The Eigen::Matrix is required
to construct an object for this class. The class is also templatized to accept
various types of data types which correspond to the data type of the values
stored in the given cost matrix. Square and rectangular cost matrices are supported.

After the object is constructed, the `minimize()` and `maximize()` functions can be
used to compute the optimal solutions. These functions return the row/column indices
as a `std::pair` of Eigen Vectors that contain the job assigment solutions from the
given cost matrix.

If the provided Eigen Matrix is the following into the `minimize()` or
`maximize()` functions:
```text
1.2f, 3.23f, 2.54f, 4.94f,
9.1f, 2.22f, 5.21f, 3.23f,
7.1f, 4.3f, 0.2f, 8.93f,
6.34f, 1.23f, 8.11f, 1.94f
```

The output will be an `std::pair` of Eigen Vectors where `std::pair::first`
will contain the assigned row indices:
```text
[0, 1, 2, 3]
```

Where the `std::pair::second` will contain the assigned column indices:
```text
[0,
 1,
 2,
 3]
```

The result means row 0 is assigned to column 0, row 1 is assigned to column 1, row 2 is
assigned to column 2, and row 3 is assigned to column 3. Therefore using these row/column
assignments to compute the total cost from the given cost matrix results in:
```text
float totalCost = 0.0f;
for (int i = 0; i < std::pair::first.size(); ++i) {
    totalCost = costMatrix(std::pair::first[i], std::pair::second[i]);

std::cout << totalCost << std::endl;
```
```text
5.56
```

Where the static `HungarianOptimizer<T>::getTotalCost()` convenience method can compute this
total cost when provided the cost matrix and assignment vectors as input.

The static `HungarianOptimizer<T>::printMatrix()` will print a cost matrix with its associated
job assignments highlighted by a prepended `*` on the corresponding cell value, when provided
the cost matrix and assignment vectors as input.

## References

1. [Apollo Hungarian Optimizer](https://github.com/ApolloAuto/apollo/blob/master/modules/perception/common/graph/hungarian_optimizer.h)
2. [Munkres’ Assignment Algorithm-Modified for Rectangular Matrices](https://brc2.com/the-algorithm-workshop/)
3. [Rochshi's Hungarian algorithm code](https://github.com/RocShi/hungarian_optimizer)
4. [Google's Hungarian algorithm code](https://github.com/google/or-tools/blob/v9.4/ortools/algorithms/hungarian.cc)
