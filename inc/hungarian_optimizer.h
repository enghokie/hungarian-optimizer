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
  std::pair<Eigen::VectorXi, Eigen::VectorXi> Maximize();

  // Like Maximize(), but minimizing the cost instead.
  std::pair<Eigen::VectorXi, Eigen::VectorXi> Minimize();

  // Print the matrix to stdout (for debugging.)
  static void PrintMatrix(
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& costs,
      const Eigen::VectorXi& rowsol, const Eigen::VectorXi& colsol);

  // Calculate the total cost given the cost matrix and optimal row/col indices.
  static T GetTotalCost(
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& cost,
      const Eigen::VectorXi& rowsol, const Eigen::VectorXi& colsol);

 private:
  typedef void (HungarianOptimizer<T>::*Step)();

  typedef enum { NONE, PRIME, STAR } Mark;

  // Convert the final cost matrix into a set of assignments of rowsol->colsol.
  // Returns the assignment in the two vectors passed as argument, the same as
  // Minimize and Maximize
  void FindAssignments(
      std::pair<Eigen::VectorXi, Eigen::VectorXi>& assignments);

  // Is the cell (row, col) starred?
  bool IsStarred(int row, int col) const { return marks_(row, col) == STAR; }

  // Mark cell (row, col) with a star
  void Star(int row, int col) {
    marks_(row, col) = STAR;
    stars_in_col_[col]++;
  }

  // Remove a star from cell (row, col)
  void UnStar(int row, int col) {
    marks_(row, col) = NONE;
    stars_in_col_[col]--;
  }

  // Find a column in row 'row' containing a star, or return
  // kHungarianOptimizerColNotFound if no such column exists.
  int FindStarInRow(int row) const;

  // Find a row in column 'col' containing a star, or return
  // kHungarianOptimizerRowNotFound if no such row exists.
  int FindStarInCol(int col) const;

  // Is cell (row, col) marked with a prime?
  bool IsPrimed(int row, int col) const { return marks_(row, col) == PRIME; }

  // Mark cell (row, col) with a prime.
  void Prime(int row, int col) { marks_(row, col) = PRIME; }

  // Find a column in row containing a prime, or return
  // kHungarianOptimizerColNotFound if no such column exists.
  int FindPrimeInRow(int row) const;

  // Remove the prime marks_ from every cell in the matrix.
  void ClearPrimes();

  // Does column col contain a star?
  bool ColContainsStar(int col) const { return stars_in_col_[col] > 0; }

  // Is row 'row' covered?
  bool RowCovered(int row) const { return rows_covered_[row]; }

  // Cover row 'row'.
  void CoverRow(int row) { rows_covered_[row] = true; }

  // Uncover row 'row'.
  void UncoverRow(int row) { rows_covered_[row] = false; }

  // Is column col covered?
  bool ColCovered(int col) const { return cols_covered_[col]; }

  // Cover column col.
  void CoverCol(int col) { cols_covered_[col] = true; }

  // Uncover column col.
  void UncoverCol(int col) { cols_covered_[col] = false; }

  // Uncover ever row and column in the matrix.
  void ClearCovers();

  // Find the smallest uncovered cell in the matrix.
  T FindSmallestUncovered() const;

  // Find an uncovered zero and store its coordinates in (zeroRow_, zeroCol_)
  // and return true, or return false if no such cell exists.
  bool FindZero(int& zero_row, int& zero_col) const;

  // Run the Munkres algorithm!
  void DoMunkres();

  // Step 1.
  // For each row of the matrix, find the smallest element and subtract it
  // from every element in its row.  Go to Step 2.
  void ReduceRows();

  // Step 2.
  // Find a zero (Z) in the matrix.  If there is no starred zero in its row
  // or column, star Z.  Repeat for every element in the matrix.  Go to step 3.
  // Note: profiling shows this method to use 9.2% of the CPU - the next
  // slowest step takes 0.6%.  I can't think of a way of speeding it up though.
  void StarZeroes();

  // Step 3.
  // Cover each column containing a starred zero.  If all columns are
  // covered, the starred zeros describe a complete set of unique assignments.
  // In this case, terminate the algorithm.  Otherwise, go to step 4.
  void CoverStarredZeroes();

  // Step 4.
  // Find a noncovered zero and prime it.  If there is no starred zero in the
  // row containing this primed zero, Go to Step 5.  Otherwise, cover this row
  // and uncover the column containing the starred zero. Continue in this manner
  // until there are no uncovered zeros left, then go to Step 6.
  void PrimeZeroes();

  // Step 5.
  // Construct a series of alternating primed and starred zeros as follows.
  // Let Z0 represent the uncovered primed zero found in Step 4.  Let Z1 denote
  // the starred zero in the column of Z0 (if any). Let Z2 denote the primed
  // zero in the row of Z1 (there will always be one).  Continue until the
  // series terminates at a primed zero that has no starred zero in its column.
  // Unstar each starred zero of the series, star each primed zero of the
  // series, erase all primes and uncover every line in the matrix.  Return to
  // Step 3.
  void MakeAugmentingPath();

  // Step 6.
  // Add the smallest uncovered value in the matrix to every element of each
  // covered row, and subtract it from every element of each uncovered column.
  // Return to Step 4 without altering any stars, primes, or covered lines.
  void AugmentPath();

  // The size of the problem, i.e. max(#agents, #tasks).
  int matrix_size_;

  // The expanded cost matrix.
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> costs_;

  // The greatest cost in the initial cost matrix.
  T max_cost_;

  // Which rows and columns are currently covered.
  std::vector<bool> rows_covered_;
  std::vector<bool> cols_covered_;

  // The marks_ (star/prime/none) on each element of the cost matrix.
  Eigen::Matrix<Mark, Eigen::Dynamic, Eigen::Dynamic> marks_;

  // The number of stars in each column - used to speed up coverStarredZeroes.
  std::vector<int> stars_in_col_;

  // Representation of a path_ through the matrix - used in step 5.
  std::vector<int> rowsol_;  // i.e. the agents
  std::vector<int> colsol_;  // i.e. the tasks

  // The width_ and height_ of the initial (non-expanded) cost matrix.
  int width_;
  int height_;

  // The current state of the algorithm
  std::function<void()> fn_state_ = nullptr;
};

template <typename T>
HungarianOptimizer<T>::HungarianOptimizer(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& costs)
    : matrix_size_(0),
      costs_(),
      max_cost_(0),
      rows_covered_(),
      cols_covered_(),
      marks_(),
      stars_in_col_(),
      rowsol_(),
      colsol_(),
      width_(0),
      height_(0) {
  width_ = costs.cols();

  if (width_ > 0) {
    height_ = costs.rows();
  } else {
    height_ = 0;
  }

  matrix_size_ = std::max(width_, height_);
  max_cost_ = 0;

  // Generate the expanded cost matrix by adding extra 0-valued elements in
  // order to make a square matrix.  At the same time, find the greatest cost
  // in the matrix (used later if we want to maximize rather than minimize the
  // overall cost.)
  costs_.resize(matrix_size_, matrix_size_);
  for (int row = 0; row < matrix_size_; ++row) {
    for (int col = 0; col < matrix_size_; ++col) {
      if ((row >= height_) || (col >= width_)) {
        costs_(row, col) = 0;
      } else {
        costs_(row, col) = costs(row, col);
        max_cost_ = std::max(max_cost_, costs_(row, col));
      }
    }
  }

  // Initially, none of the cells of the matrix are marked.
  marks_.resize(matrix_size_, matrix_size_);
  for (int row = 0; row < matrix_size_; ++row) {
    for (int col = 0; col < matrix_size_; ++col) {
      marks_(row, col) = NONE;
    }
  }

  rows_covered_.resize(matrix_size_, false);
  cols_covered_.resize(matrix_size_, false);
  stars_in_col_.resize(matrix_size_, 0);
  rowsol_.resize(matrix_size_ * 2, 0);
  colsol_.resize(matrix_size_ * 2, 0);
}

// Find an assignment which maximizes the total cost.
// Return an array of pairs of integers.  Each pair (i, j) corresponds to
// assigning agent i to task j.
template <typename T>
std::pair<Eigen::VectorXi, Eigen::VectorXi> HungarianOptimizer<T>::Maximize() {
  // Find a maximal assignment by subtracting each of the
  // original costs from max_cost_  and then minimizing.
  for (int row = 0; row < height_; ++row) {
    for (int col = 0; col < width_; ++col) {
      costs_(row, col) = max_cost_ - costs_(row, col);
    }
  }

  return Minimize();
}

// Find an assignment which minimizes the total cost.
// Return an array of pairs of integers.  Each pair (i, j) corresponds to
// assigning agent i to task j.
template <typename T>
std::pair<Eigen::VectorXi, Eigen::VectorXi> HungarianOptimizer<T>::Minimize() {
  DoMunkres();

  std::pair<Eigen::VectorXi, Eigen::VectorXi> assignments;
  FindAssignments(assignments);

  return assignments;
}

// Convert the final cost matrix into a set of assignments of agents -> tasks.
// Return an array of pairs of integers, the same as the return values of
// Minimize() and Maximize()
template <typename T>
void HungarianOptimizer<T>::FindAssignments(
    std::pair<Eigen::VectorXi, Eigen::VectorXi>& assignments) {
  auto minDims = std::min(width_, height_);
  assignments.first.resize(minDims);
  assignments.second.resize(minDims);

  auto idx = 0;
  for (int row = 0; row < height_; ++row) {
    for (int col = 0; col < width_; ++col) {
      if (IsStarred(row, col)) {
        assignments.first[idx] = row;
        assignments.second[idx] = col;
        ++idx;
        break;
      }
    }
  }
  // TODO(user)
  // result_size = min(width_, height_);
  // CHECK colsol.size() == result_size
  // CHECK rowsol.size() == result_size
}

// Find a column in row 'row' containing a star, or return
// kHungarianOptimizerColNotFound if no such column exists.
template <typename T>
int HungarianOptimizer<T>::FindStarInRow(int row) const {
  for (int col = 0; col < matrix_size_; ++col) {
    if (IsStarred(row, col)) {
      return col;
    }
  }

  return kHungarianOptimizerColNotFound;
}

// Find a row in column 'col' containing a star, or return
// kHungarianOptimizerRowNotFound if no such row exists.
template <typename T>
int HungarianOptimizer<T>::FindStarInCol(int col) const {
  if (!ColContainsStar(col)) {
    return kHungarianOptimizerRowNotFound;
  }

  for (int row = 0; row < matrix_size_; ++row) {
    if (IsStarred(row, col)) {
      return row;
    }
  }

  // NOTREACHED
  return kHungarianOptimizerRowNotFound;
}

// Find a column in row containing a prime, or return
// kHungarianOptimizerColNotFound if no such column exists.
template <typename T>
int HungarianOptimizer<T>::FindPrimeInRow(int row) const {
  for (int col = 0; col < matrix_size_; ++col) {
    if (IsPrimed(row, col)) {
      return col;
    }
  }

  return kHungarianOptimizerColNotFound;
}

// Remove the prime marks from every cell in the matrix.
template <typename T>
void HungarianOptimizer<T>::ClearPrimes() {
  for (int row = 0; row < matrix_size_; ++row) {
    for (int col = 0; col < matrix_size_; ++col) {
      if (IsPrimed(row, col)) {
        marks_(row, col) = NONE;
      }
    }
  }
}

// Uncovery ever row and column in the matrix.
template <typename T>
void HungarianOptimizer<T>::ClearCovers() {
  for (int x = 0; x < matrix_size_; x++) {
    UncoverRow(x);
    UncoverCol(x);
  }
}

// Find the smallest uncovered cell in the matrix.
template <typename T>
T HungarianOptimizer<T>::FindSmallestUncovered() const {
  T minval = std::numeric_limits<T>::max();

  for (int row = 0; row < matrix_size_; ++row) {
    if (RowCovered(row)) {
      continue;
    }

    for (int col = 0; col < matrix_size_; ++col) {
      if (ColCovered(col)) {
        continue;
      }

      minval = std::min(minval, costs_(row, col));
    }
  }

  return minval;
}

// Find an uncovered zero and store its co-ordinates in (zeroRow, zeroCol)
// and return true, or return false if no such cell exists.
template <typename T>
bool HungarianOptimizer<T>::FindZero(int& zero_row, int& zero_col) const {
  for (int row = 0; row < matrix_size_; ++row) {
    if (RowCovered(row)) {
      continue;
    }

    for (int col = 0; col < matrix_size_; ++col) {
      if (ColCovered(col)) {
        continue;
      }

      if (costs_(row, col) == 0) {
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
void HungarianOptimizer<T>::PrintMatrix(
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
T HungarianOptimizer<T>::GetTotalCost(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& cost,
    const Eigen::VectorXi& rowsol, const Eigen::VectorXi& colsol) {
  T totalCost = 0;
  auto minDims = std::min(rowsol.size(), colsol.size());
  for (int i = 0; i < minDims; ++i) totalCost += cost(rowsol[i], colsol[i]);

  return totalCost;
}

//  Run the Munkres algorithm!
template <typename T>
void HungarianOptimizer<T>::DoMunkres() {
  fn_state_ = std::bind(&HungarianOptimizer<T>::ReduceRows, this);
  while (fn_state_ != nullptr) {
    fn_state_();
  }
}

// Step 1.
// For each row of the matrix, find the smallest element and subtract it
// from every element in its row.  Go to Step 2.
template <typename T>
void HungarianOptimizer<T>::ReduceRows() {
  for (int row = 0; row < matrix_size_; ++row) {
    T min_cost = costs_(row, 0);
    for (int col = 1; col < matrix_size_; ++col) {
      min_cost = std::min(min_cost, costs_(row, col));
    }
    for (int col = 0; col < matrix_size_; ++col) {
      costs_(row, col) -= min_cost;
    }
  }
  fn_state_ = std::bind(&HungarianOptimizer<T>::StarZeroes, this);
}

// Step 2.
// Find a zero (Z) in the matrix.  If there is no starred zero in its row
// or column, star Z.  Repeat for every element in the matrix.  Go to step 3.
template <typename T>
void HungarianOptimizer<T>::StarZeroes() {
  // Since no rows or columns are covered on entry to this step, we use the
  // covers as a quick way of marking which rows & columns have stars in them.
  for (int row = 0; row < matrix_size_; ++row) {
    if (RowCovered(row)) {
      continue;
    }

    for (int col = 0; col < matrix_size_; ++col) {
      if (ColCovered(col)) {
        continue;
      }

      if (costs_(row, col) == 0) {
        Star(row, col);
        CoverRow(row);
        CoverCol(col);
        break;
      }
    }
  }

  ClearCovers();
  fn_state_ = std::bind(&HungarianOptimizer<T>::CoverStarredZeroes, this);
}

// Step 3.
// Cover each column containing a starred zero.  If all columns are
// covered, the starred zeros describe a complete set of unique assignments.
// In this case, terminate the algorithm.  Otherwise, go to step 4.
template <typename T>
void HungarianOptimizer<T>::CoverStarredZeroes() {
  int num_covered = 0;

  for (int col = 0; col < matrix_size_; ++col) {
    if (ColContainsStar(col)) {
      CoverCol(col);
      num_covered++;
    }
  }

  if (num_covered >= matrix_size_) {
    fn_state_ = nullptr;
    return;
  }
  fn_state_ = std::bind(&HungarianOptimizer<T>::PrimeZeroes, this);
}

// Step 4.
// Find a noncovered zero and prime it.  If there is no starred zero in the
// row containing this primed zero, Go to Step 5.  Otherwise, cover this row
// and uncover the column containing the starred zero. Continue in this manner
// until there are no uncovered zeros left, then go to Step 6.
template <typename T>
void HungarianOptimizer<T>::PrimeZeroes() {
  // This loop is guaranteed to terminate in at most matrix_size_ iterations,
  // as findZero() returns a location only if there is at least one uncovered
  // zero in the matrix.  Each iteration, either one row is covered or the
  // loop terminates.  Since there are matrix_size_ rows, after that many
  // iterations there are no uncovered cells and hence no uncovered zeroes,
  // so the loop terminates.
  for (;;) {
    int zero_row, zero_col;
    if (!FindZero(zero_row, zero_col)) {
      // No uncovered zeroes.
      fn_state_ = std::bind(&HungarianOptimizer<T>::AugmentPath, this);
      return;
    }

    Prime(zero_row, zero_col);
    int star_col = FindStarInRow(zero_row);

    if (star_col != kHungarianOptimizerColNotFound) {
      CoverRow(zero_row);
      UncoverCol(star_col);
    } else {
      rowsol_[0] = zero_row;
      colsol_[0] = zero_col;
      fn_state_ = std::bind(&HungarianOptimizer<T>::MakeAugmentingPath, this);
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
void HungarianOptimizer<T>::MakeAugmentingPath() {
  bool done = false;
  int count = 0;

  // Note: this loop is guaranteed to terminate within matrix_size_ iterations
  // because:
  // 1) on entry to this step, there is at least 1 column with no starred zero
  //    (otherwise we would have terminated the algorithm already.)
  // 2) each row containing a star also contains exactly one primed zero.
  // 4) each column contains at most one starred zero.
  //
  // Since the path_ we construct visits primed and starred zeroes alternately,
  // and terminates if we reach a primed zero in a column with no star, our
  // path_ must either contain matrix_size_ or fewer stars (in which case the
  // loop iterates fewer than matrix_size_ times), or it contains more.  In
  // that case, because (1) implies that there are fewer than
  // matrix_size_ stars, we must have visited at least one star more than once.
  // Consider the first such star that we visit more than once; it must have
  // been reached immediately after visiting a prime in the same row.  By (2),
  // this prime is unique and so must have also been visited more than once.
  // Therefore, that prime must be in the same column as a star that has been
  // visited more than once, contradicting the assumption that we chose the
  // first multiply visited star, or it must be in the same column as more
  // than one star, contradicting (3).  Therefore, we never visit any star
  // more than once and the loop terminates within matrix_size_ iterations.

  while (!done) {
    // First construct the alternating path...
    int row = FindStarInCol(colsol_[count]);

    if (row != kHungarianOptimizerRowNotFound) {
      count++;
      rowsol_[count] = row;
      colsol_[count] = colsol_[count - 1];
    } else {
      done = true;
    }

    if (!done) {
      int col = FindPrimeInRow(rowsol_[count]);
      count++;
      rowsol_[count] = rowsol_[count - 1];
      colsol_[count] = col;
    }
  }

  // Then modify it.
  for (int i = 0; i <= count; ++i) {
    int row = rowsol_[i];
    int col = colsol_[i];

    if (IsStarred(row, col)) {
      UnStar(row, col);
    } else {
      Star(row, col);
    }
  }

  ClearCovers();
  ClearPrimes();
  fn_state_ = std::bind(&HungarianOptimizer<T>::CoverStarredZeroes, this);
}

// Step 6
// Add the smallest uncovered value in the matrix to every element of each
// covered row, and subtract it from every element of each uncovered column.
// Return to Step 4 without altering any stars, primes, or covered lines.
template <typename T>
void HungarianOptimizer<T>::AugmentPath() {
  T minval = FindSmallestUncovered();

  for (int row = 0; row < matrix_size_; ++row) {
    for (int col = 0; col < matrix_size_; ++col) {
      if (RowCovered(row)) {
        costs_(row, col) += minval;
      }

      if (!ColCovered(col)) {
        costs_(row, col) -= minval;
      }
    }
  }

  fn_state_ = std::bind(&HungarianOptimizer<T>::PrimeZeroes, this);
}
