// Copyright 2010-2022 Google LLC
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//
// IMPORTANT NOTE: we advise using the code in
// graph/linear_assignment.h whose complexity is
// usually much smaller.
// TODO(user): base this code on LinearSumAssignment.
//
// For each of the four functions declared in this file, in case the input
// parameter 'cost' contains NaN, the function will return without invoking the
// Hungarian algorithm, and the output parameters 'direct_assignment' and
// 'reverse_assignment' will be left unchanged.
//

// An O(n^4) implementation of the Kuhn-Munkres algorithm (a.k.a. the
// Hungarian algorithm) for solving the assignment problem.
// The assignment problem takes a set of agents, a set of tasks and a
// cost associated with assigning each agent to each task and produces
// an optimal (i.e., least cost) assignment of agents to tasks.
// The code also enables computing a maximum assignment by changing the
// input matrix.
//
// This code is based on (read: translated from) the Java version
// (read: translated from) the Python version at
//   http://www.clapper.org/software/python/munkres/.

#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <vector>

template <typename T>
class HungarianOptimizer {
  static constexpr int kHungarianOptimizerRowNotFound = -1;
  static constexpr int kHungarianOptimizerColNotFound = -2;

 public:
  // Setup the initial conditions for the algorithm.

  // Parameters: costs is a matrix of the cost of assigning each agent to
  // each task. costs[i][j] is the cost of assigning agent i to task j.
  // All the costs must be non-negative.  This matrix does not have to
  // be square (i.e. we can have different numbers of agents and tasks), but it
  // must be regular (i.e. there must be the same number of entries in each row
  // of the matrix).
  explicit HungarianOptimizer(
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& costs);

  // Find an assignment which maximizes the total cost.
  // Returns the assignment in the two vectors passed as argument.
  // rowsol[i] is assigned to colsol[i].
  std::pair<Eigen::VectorXi, Eigen::VectorXi> maximize();

  // Like maximize(), but minimizing the cost instead.
  std::pair<Eigen::VectorXi, Eigen::VectorXi> minimize();

  // Print the matrix to stdout (for debugging.)
  static void printMatrix(
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& costs,
      const Eigen::VectorXi& rowsol, const Eigen::VectorXi& colsol);

  // Calculate the total cost given the cost matrix and optimal row/col indices.
  static T getTotalCost(
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& cost,
      const Eigen::VectorXi& rowsol, const Eigen::VectorXi& colsol);

 private:
  typedef void (HungarianOptimizer<T>::*Step)();

  enum class Mark { NONE, PRIME, STAR };

  // Convert the final cost matrix into a set of assignments of rowsol->colsol.
  // Returns the assignment in the two vectors passed as argument, the same as
  // Minimize and Maximize
  void findAssignments(
      std::pair<Eigen::VectorXi, Eigen::VectorXi>& assignments);

  // Is the cell (row, col) starred?
  bool isStarred(int row, int col) const {
    return _marks(row, col) == Mark::STAR;
  }

  // Mark cell (row, col) with a star
  void star(int row, int col) {
    _marks(row, col) = Mark::STAR;
    _starsInCol[col]++;
  }

  // Remove a star from cell (row, col)
  void unstar(int row, int col) {
    _marks(row, col) = Mark::NONE;
    _starsInCol[col]--;
  }

  // Find a column in row 'row' containing a star, or return
  // kHungarianOptimizerColNotFound if no such column exists.
  int findStarInRow(int row) const;

  // Find a row in column 'col' containing a star, or return
  // kHungarianOptimizerRowNotFound if no such row exists.
  int findStarInCol(int col) const;

  // Is cell (row, col) marked with a prime?
  bool isPrimed(int row, int col) const {
    return _marks(row, col) == Mark::PRIME;
  }

  // Mark cell (row, col) with a prime.
  void prime(int row, int col) { _marks(row, col) = Mark::PRIME; }

  // Find a column in row containing a prime, or return
  // kHungarianOptimizerColNotFound if no such column exists.
  int findPrimeInRow(int row) const;

  // Remove the prime _marks from every cell in the matrix.
  void clearPrimes();

  // Does column col contain a star?
  bool colContainsStar(int col) const { return _starsInCol[col] > 0; }

  // Is row 'row' covered?
  bool rowCovered(int row) const { return _rowsCovered[row]; }

  // Cover row 'row'.
  void coverRow(int row) { _rowsCovered[row] = true; }

  // Uncover row 'row'.
  void uncoverRow(int row) { _rowsCovered[row] = false; }

  // Is column col covered?
  bool colCovered(int col) const { return _colsCovered[col]; }

  // Cover column col.
  void coverCol(int col) { _colsCovered[col] = true; }

  // Uncover column col.
  void uncoverCol(int col) { _colsCovered[col] = false; }

  // Uncover ever row and column in the matrix.
  void clearCovers();

  // Find the smallest uncovered cell in the matrix.
  T findSmallestUncovered() const;

  // Find an uncovered zero and store its coordinates in (zeroRow_, zeroCol_)
  // and return true, or return false if no such cell exists.
  bool findZero(int& zero_row, int& zero_col) const;

  // Run the Munkres algorithm!
  void doMunkres();

  // Step 1.
  // For each row of the matrix, find the smallest element and subtract it
  // from every element in its row.  Go to Step 2.
  void reduceRows();

  // Step 2.
  // Find a zero (Z) in the matrix.  If there is no starred zero in its row
  // or column, star Z.  Repeat for every element in the matrix.  Go to step 3.
  // Note: profiling shows this method to use 9.2% of the CPU - the next
  // slowest step takes 0.6%.  I can't think of a way of speeding it up though.
  void starZeroes();

  // Step 3.
  // Cover each column containing a starred zero.  If all columns are
  // covered, the starred zeros describe a complete set of unique assignments.
  // In this case, terminate the algorithm.  Otherwise, go to step 4.
  void coverStarredZeroes();

  // Step 4.
  // Find a noncovered zero and prime it.  If there is no starred zero in the
  // row containing this primed zero, Go to Step 5.  Otherwise, cover this row
  // and uncover the column containing the starred zero. Continue in this manner
  // until there are no uncovered zeros left, then go to Step 6.
  void primeZeroes();

  // Step 5.
  // Construct a series of alternating primed and starred zeros as follows.
  // Let Z0 represent the uncovered primed zero found in Step 4.  Let Z1 denote
  // the starred zero in the column of Z0 (if any). Let Z2 denote the primed
  // zero in the row of Z1 (there will always be one).  Continue until the
  // series terminates at a primed zero that has no starred zero in its column.
  // Unstar each starred zero of the series, star each primed zero of the
  // series, erase all primes and uncover every line in the matrix.  Return to
  // Step 3.
  void makeAugmentingPath();

  // Step 6.
  // Add the smallest uncovered value in the matrix to every element of each
  // covered row, and subtract it from every element of each uncovered column.
  // Return to Step 4 without altering any stars, primes, or covered lines.
  void augmentPath();

  // The size of the problem, i.e. max(#agents, #tasks).
  int _matrixSize;

  // The expanded cost matrix.
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _costs;

  // The greatest cost in the initial cost matrix.
  T _maxCost;

  // Which rows and columns are currently covered.
  std::vector<bool> _rowsCovered;
  std::vector<bool> _colsCovered;

  // The _marks (star/prime/none) on each element of the cost matrix.
  Eigen::Matrix<Mark, Eigen::Dynamic, Eigen::Dynamic> _marks;

  // The number of stars in each column - used to speed up coverStarredZeroes.
  std::vector<int> _starsInCol;

  // Representation of a path_ through the matrix - used in step 5.
  std::vector<int> _rowsol;  // i.e. the agents
  std::vector<int> _colsol;  // i.e. the tasks

  // The _width and _height of the initial (non-expanded) cost matrix.
  int _width;
  int _height;

  // The current state of the algorithm
  std::function<void()> _fnState = nullptr;
};

template <typename T>
HungarianOptimizer<T>::HungarianOptimizer(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& costs)
    : _matrixSize(0),
      _costs(),
      _maxCost(0),
      _rowsCovered(),
      _colsCovered(),
      _marks(),
      _starsInCol(),
      _rowsol(),
      _colsol(),
      _width(0),
      _height(0) {
  _width = costs.cols();

  if (_width > 0) {
    _height = costs.rows();
  } else {
    _height = 0;
  }

  _matrixSize = std::max(_width, _height);
  _maxCost = 0;

  // Generate the expanded cost matrix by adding extra 0-valued elements in
  // order to make a square matrix.  At the same time, find the greatest cost
  // in the matrix (used later if we want to maximize rather than minimize the
  // overall cost.)
  _costs.resize(_matrixSize, _matrixSize);
  for (int row = 0; row < _matrixSize; ++row) {
    for (int col = 0; col < _matrixSize; ++col) {
      if ((row >= _height) || (col >= _width)) {
        _costs(row, col) = 0;
      } else {
        _costs(row, col) = costs(row, col);
        _maxCost = std::max(_maxCost, _costs(row, col));
      }
    }
  }

  // Initially, none of the cells of the matrix are marked.
  _marks.resize(_matrixSize, _matrixSize);
  for (int row = 0; row < _matrixSize; ++row) {
    for (int col = 0; col < _matrixSize; ++col) {
      _marks(row, col) = Mark::NONE;
    }
  }

  _rowsCovered.resize(_matrixSize, false);
  _colsCovered.resize(_matrixSize, false);
  _starsInCol.resize(_matrixSize, 0);
  _rowsol.resize(_matrixSize * 2, 0);
  _colsol.resize(_matrixSize * 2, 0);
}

// Find an assignment which maximizes the total cost.
// Return an array of pairs of integers.  Each pair (i, j) corresponds to
// assigning agent i to task j.
template <typename T>
std::pair<Eigen::VectorXi, Eigen::VectorXi> HungarianOptimizer<T>::maximize() {
  // Find a maximal assignment by subtracting each of the
  // original costs from _maxCost  and then minimizing.
  for (int row = 0; row < _height; ++row) {
    for (int col = 0; col < _width; ++col) {
      _costs(row, col) = _maxCost - _costs(row, col);
    }
  }

  return minimize();
}

// Find an assignment which minimizes the total cost.
// Return an array of pairs of integers.  Each pair (i, j) corresponds to
// assigning agent i to task j.
template <typename T>
std::pair<Eigen::VectorXi, Eigen::VectorXi> HungarianOptimizer<T>::minimize() {
  doMunkres();

  std::pair<Eigen::VectorXi, Eigen::VectorXi> assignments;
  findAssignments(assignments);

  return assignments;
}

// Convert the final cost matrix into a set of assignments of agents -> tasks.
// Return an array of pairs of integers, the same as the return values of
// minimize() and maximize()
template <typename T>
void HungarianOptimizer<T>::findAssignments(
    std::pair<Eigen::VectorXi, Eigen::VectorXi>& assignments) {
  auto minDims = std::min(_width, _height);
  assignments.first.resize(minDims);
  assignments.second.resize(minDims);

  auto idx = 0;
  for (int row = 0; row < _height; ++row) {
    for (int col = 0; col < _width; ++col) {
      if (isStarred(row, col)) {
        assignments.first[idx] = row;
        assignments.second[idx] = col;
        ++idx;
        break;
      }
    }
  }
  // TODO(user)
  // result_size = min(_width, _height);
  // CHECK colsol.size() == result_size
  // CHECK rowsol.size() == result_size
}

// Find a column in row 'row' containing a star, or return
// kHungarianOptimizerColNotFound if no such column exists.
template <typename T>
int HungarianOptimizer<T>::findStarInRow(int row) const {
  for (int col = 0; col < _matrixSize; ++col) {
    if (isStarred(row, col)) {
      return col;
    }
  }

  return kHungarianOptimizerColNotFound;
}

// Find a row in column 'col' containing a star, or return
// kHungarianOptimizerRowNotFound if no such row exists.
template <typename T>
int HungarianOptimizer<T>::findStarInCol(int col) const {
  if (!colContainsStar(col)) {
    return kHungarianOptimizerRowNotFound;
  }

  for (int row = 0; row < _matrixSize; ++row) {
    if (isStarred(row, col)) {
      return row;
    }
  }

  // NOTREACHED
  return kHungarianOptimizerRowNotFound;
}

// Find a column in row containing a prime, or return
// kHungarianOptimizerColNotFound if no such column exists.
template <typename T>
int HungarianOptimizer<T>::findPrimeInRow(int row) const {
  for (int col = 0; col < _matrixSize; ++col) {
    if (isPrimed(row, col)) {
      return col;
    }
  }

  return kHungarianOptimizerColNotFound;
}

// Remove the prime marks from every cell in the matrix.
template <typename T>
void HungarianOptimizer<T>::clearPrimes() {
  for (int row = 0; row < _matrixSize; ++row) {
    for (int col = 0; col < _matrixSize; ++col) {
      if (isPrimed(row, col)) {
        _marks(row, col) = Mark::NONE;
      }
    }
  }
}

// Uncovery ever row and column in the matrix.
template <typename T>
void HungarianOptimizer<T>::clearCovers() {
  for (int x = 0; x < _matrixSize; x++) {
    uncoverRow(x);
    uncoverCol(x);
  }
}

// Find the smallest uncovered cell in the matrix.
template <typename T>
T HungarianOptimizer<T>::findSmallestUncovered() const {
  T minval = std::numeric_limits<T>::max();

  for (int row = 0; row < _matrixSize; ++row) {
    if (rowCovered(row)) {
      continue;
    }

    for (int col = 0; col < _matrixSize; ++col) {
      if (colCovered(col)) {
        continue;
      }

      minval = std::min(minval, _costs(row, col));
    }
  }

  return minval;
}

// Find an uncovered zero and store its co-ordinates in (zeroRow, zeroCol)
// and return true, or return false if no such cell exists.
template <typename T>
bool HungarianOptimizer<T>::findZero(int& zero_row, int& zero_col) const {
  for (int row = 0; row < _matrixSize; ++row) {
    if (rowCovered(row)) {
      continue;
    }

    for (int col = 0; col < _matrixSize; ++col) {
      if (colCovered(col)) {
        continue;
      }

      if (_costs(row, col) == 0) {
        zero_row = row;
        zero_col = col;
        return true;
      }
    }
  }

  return false;
}

// Print the matrix to stdout (for debugging.)
template <typename T>
void HungarianOptimizer<T>::printMatrix(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& costs,
    const Eigen::VectorXi& rowsol, const Eigen::VectorXi& colsol) {
  auto minDims = std::min(rowsol.size(), colsol.size());
  for (int row = 0; row < costs.rows(); ++row) {
    for (int col = 0; col < costs.cols(); ++col) {
      for (int i = 0; i < minDims; ++i)
        if (rowsol[i] == row && colsol[i] == col) printf("*");

      printf("%g ", costs(row, col));
    }
    printf("\n");
  }
}

// @brief Computes the total cost from the given assignments are cost matrix.
// @param cost The cost matrix to compute the total cost from.
// @param assignments Cost assignments to get the total cost from.
template <typename T>
T HungarianOptimizer<T>::getTotalCost(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& cost,
    const Eigen::VectorXi& rowsol, const Eigen::VectorXi& colsol) {
  T totalCost = 0;
  auto minDims = std::min(rowsol.size(), colsol.size());
  for (int i = 0; i < minDims; ++i) totalCost += cost(rowsol[i], colsol[i]);

  return totalCost;
}

//  Run the Munkres algorithm!
template <typename T>
void HungarianOptimizer<T>::doMunkres() {
  _fnState = std::bind(&HungarianOptimizer<T>::reduceRows, this);
  while (_fnState != nullptr) {
    _fnState();
  }
}

// Step 1.
// For each row of the matrix, find the smallest element and subtract it
// from every element in its row.  Go to Step 2.
template <typename T>
void HungarianOptimizer<T>::reduceRows() {
  for (int row = 0; row < _matrixSize; ++row) {
    T min_cost = _costs(row, 0);
    for (int col = 1; col < _matrixSize; ++col) {
      min_cost = std::min(min_cost, _costs(row, col));
    }
    for (int col = 0; col < _matrixSize; ++col) {
      _costs(row, col) -= min_cost;
    }
  }
  _fnState = std::bind(&HungarianOptimizer<T>::starZeroes, this);
}

// Step 2.
// Find a zero (Z) in the matrix.  If there is no starred zero in its row
// or column, star Z.  Repeat for every element in the matrix.  Go to step 3.
template <typename T>
void HungarianOptimizer<T>::starZeroes() {
  // Since no rows or columns are covered on entry to this step, we use the
  // covers as a quick way of marking which rows & columns have stars in them.
  for (int row = 0; row < _matrixSize; ++row) {
    if (rowCovered(row)) {
      continue;
    }

    for (int col = 0; col < _matrixSize; ++col) {
      if (colCovered(col)) {
        continue;
      }

      if (_costs(row, col) == 0) {
        star(row, col);
        coverRow(row);
        coverCol(col);
        break;
      }
    }
  }

  clearCovers();
  _fnState = std::bind(&HungarianOptimizer<T>::coverStarredZeroes, this);
}

// Step 3.
// Cover each column containing a starred zero.  If all columns are
// covered, the starred zeros describe a complete set of unique assignments.
// In this case, terminate the algorithm.  Otherwise, go to step 4.
template <typename T>
void HungarianOptimizer<T>::coverStarredZeroes() {
  int num_covered = 0;

  for (int col = 0; col < _matrixSize; ++col) {
    if (colContainsStar(col)) {
      coverCol(col);
      num_covered++;
    }
  }

  if (num_covered >= _matrixSize) {
    _fnState = nullptr;
    return;
  }
  _fnState = std::bind(&HungarianOptimizer<T>::primeZeroes, this);
}

// Step 4.
// Find a noncovered zero and prime it.  If there is no starred zero in the
// row containing this primed zero, Go to Step 5.  Otherwise, cover this row
// and uncover the column containing the starred zero. Continue in this manner
// until there are no uncovered zeros left, then go to Step 6.
template <typename T>
void HungarianOptimizer<T>::primeZeroes() {
  // This loop is guaranteed to terminate in at most _matrixSize iterations,
  // as findZero() returns a location only if there is at least one uncovered
  // zero in the matrix.  Each iteration, either one row is covered or the
  // loop terminates.  Since there are _matrixSize rows, after that many
  // iterations there are no uncovered cells and hence no uncovered zeroes,
  // so the loop terminates.
  for (;;) {
    int zero_row, zero_col;
    if (!findZero(zero_row, zero_col)) {
      // No uncovered zeroes.
      _fnState = std::bind(&HungarianOptimizer<T>::augmentPath, this);
      return;
    }

    prime(zero_row, zero_col);
    int star_col = findStarInRow(zero_row);

    if (star_col != kHungarianOptimizerColNotFound) {
      coverRow(zero_row);
      uncoverCol(star_col);
    } else {
      _rowsol[0] = zero_row;
      _colsol[0] = zero_col;
      _fnState = std::bind(&HungarianOptimizer<T>::makeAugmentingPath, this);
      return;
    }
  }
}

// Step 5.
// Construct a series of alternating primed and starred zeros as follows.
// Let Z0 represent the uncovered primed zero found in Step 4.  Let Z1 denote
// the starred zero in the column of Z0 (if any). Let Z2 denote the primed
// zero in the row of Z1 (there will always be one).  Continue until the
// series terminates at a primed zero that has no starred zero in its column.
// Unstar each starred zero of the series, star each primed zero of the
// series, erase all primes and uncover every line in the matrix.  Return to
// Step 3.
template <typename T>
void HungarianOptimizer<T>::makeAugmentingPath() {
  bool done = false;
  int count = 0;

  // Note: this loop is guaranteed to terminate within _matrixSize iterations
  // because:
  // 1) on entry to this step, there is at least 1 column with no starred zero
  //    (otherwise we would have terminated the algorithm already.)
  // 2) each row containing a star also contains exactly one primed zero.
  // 4) each column contains at most one starred zero.
  //
  // Since the path_ we construct visits primed and starred zeroes alternately,
  // and terminates if we reach a primed zero in a column with no star, our
  // path_ must either contain _matrixSize or fewer stars (in which case the
  // loop iterates fewer than _matrixSize times), or it contains more.  In
  // that case, because (1) implies that there are fewer than
  // _matrixSize stars, we must have visited at least one star more than once.
  // Consider the first such star that we visit more than once; it must have
  // been reached immediately after visiting a prime in the same row.  By (2),
  // this prime is unique and so must have also been visited more than once.
  // Therefore, that prime must be in the same column as a star that has been
  // visited more than once, contradicting the assumption that we chose the
  // first multiply visited star, or it must be in the same column as more
  // than one star, contradicting (3).  Therefore, we never visit any star
  // more than once and the loop terminates within _matrixSize iterations.

  while (!done) {
    // First construct the alternating path...
    int row = findStarInCol(_colsol[count]);

    if (row != kHungarianOptimizerRowNotFound) {
      count++;
      _rowsol[count] = row;
      _colsol[count] = _colsol[count - 1];
    } else {
      done = true;
    }

    if (!done) {
      int col = findPrimeInRow(_rowsol[count]);
      count++;
      _rowsol[count] = _rowsol[count - 1];
      _colsol[count] = col;
    }
  }

  // Then modify it.
  for (int i = 0; i <= count; ++i) {
    int row = _rowsol[i];
    int col = _colsol[i];

    if (isStarred(row, col)) {
      unstar(row, col);
    } else {
      star(row, col);
    }
  }

  clearCovers();
  clearPrimes();
  _fnState = std::bind(&HungarianOptimizer<T>::coverStarredZeroes, this);
}

// Step 6
// Add the smallest uncovered value in the matrix to every element of each
// covered row, and subtract it from every element of each uncovered column.
// Return to Step 4 without altering any stars, primes, or covered lines.
template <typename T>
void HungarianOptimizer<T>::augmentPath() {
  T minval = findSmallestUncovered();

  for (int row = 0; row < _matrixSize; ++row) {
    for (int col = 0; col < _matrixSize; ++col) {
      if (rowCovered(row)) {
        _costs(row, col) += minval;
      }

      if (!colCovered(col)) {
        _costs(row, col) -= minval;
      }
    }
  }

  _fnState = std::bind(&HungarianOptimizer<T>::primeZeroes, this);
}
