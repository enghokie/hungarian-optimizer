#include <doctest.h>
DOCTEST_MAKE_STD_HEADERS_CLEAN_FROM_WARNINGS_ON_WALL_BEGIN
#include <chrono>
#include <iostream>
#include <random>
DOCTEST_MAKE_STD_HEADERS_CLEAN_FROM_WARNINGS_ON_WALL_END

#include "hungarian_optimizer.h"

TEST_SUITE("Test Hungarian algorithm accuracy") {
  TEST_CASE("Test 4x4 cost matrix") {
    SUBCASE("Optimal diagonal solution") {
      MESSAGE("Test a 4x4 cost matrix with an optimal diagonal solution");
      int dim = 4;
      Eigen::MatrixXd costMatrix(dim, dim);
      costMatrix << 1.2, 3.23, 2.54, 4.94, 9.1, 2.22, 5.21, 3.23, 7.1, 4.3, 0.2,
          8.93, 6.34, 1.23, 8.11, 1.94;

      HungarianOptimizer<double> optimizer(costMatrix);
      const auto assignments = optimizer.minimize();
      const auto& rowsol = assignments.first;
      const auto& colsol = assignments.second;
      const auto totalCost =
          HungarianOptimizer<double>::getTotalCost(costMatrix, rowsol, colsol);
      HungarianOptimizer<double>::printMatrix(costMatrix, rowsol, colsol);

      // Check expected values
      auto expectedCost = 5.5600000000000005d;
      CHECK_EQ(int((totalCost - expectedCost) * 10000000000000000), 0);
      Eigen::VectorXi expRowsol(dim);
      expRowsol << 0, 1, 2, 3;
      Eigen::VectorXi expColsol(dim);
      expColsol << 0, 1, 2, 3;
      CHECK_EQ(rowsol, expRowsol);
      CHECK_EQ(colsol, expColsol);
    }

    SUBCASE("Optimal reversed diagonal solution") {
      MESSAGE(
          "Test a 4x4 cost matrix with an optimal reversed diagonal solution");
      int dim = 4;
      Eigen::MatrixXd costMatrix(dim, dim);
      costMatrix << 4.94, 3.23, 2.54, 1.2, 9.1, 5.21, 2.22, 3.23, 7.1, 0.2, 4.3,
          8.93, 1.94, 1.23, 8.11, 6.34;

      HungarianOptimizer<double> optimizer(costMatrix);
      const auto assignments = optimizer.minimize();
      const auto& rowsol = assignments.first;
      const auto& colsol = assignments.second;
      const auto totalCost =
          HungarianOptimizer<double>::getTotalCost(costMatrix, rowsol, colsol);
      HungarianOptimizer<double>::printMatrix(costMatrix, rowsol, colsol);

      // Check expected values
      auto expectedCost = 5.5600000000000005d;
      CHECK_EQ(int((totalCost - expectedCost) * 10000000000000000), 0);
      Eigen::VectorXi expRowsol(dim);
      expRowsol << 0, 1, 2, 3;
      Eigen::VectorXi expColsol(dim);
      expColsol << 3, 2, 1, 0;
      CHECK_EQ(rowsol, expRowsol);
      CHECK_EQ(colsol, expColsol);
    }

    SUBCASE("All decimal values") {
      MESSAGE("Test a 4x4 cost matrix with all decimal values");
      int dim = 4;
      Eigen::MatrixXd costMatrix(dim, dim);
      costMatrix << 0.2, 0.23, 0.54, 0.2, 0.1, 0.2, 0.22, 0.23, 0.3, 0.2, 0.2,
          0.93, 0.94, 0.23, 0.11, 0.94;

      HungarianOptimizer<double> optimizer(costMatrix);
      const auto assignments = optimizer.minimize();
      const auto& rowsol = assignments.first;
      const auto& colsol = assignments.second;
      const auto totalCost =
          HungarianOptimizer<double>::getTotalCost(costMatrix, rowsol, colsol);
      HungarianOptimizer<double>::printMatrix(costMatrix, rowsol, colsol);

      // Check expected values
      auto expectedCost = 0.61d;
      CHECK_EQ(int((totalCost - expectedCost) * 100), 0);
      Eigen::VectorXi expRowsol(dim);
      expRowsol << 0, 1, 2, 3;
      Eigen::VectorXi expColsol(dim);
      expColsol << 3, 0, 1, 2;
      CHECK_EQ(rowsol, expRowsol);
      CHECK_EQ(colsol, expColsol);
    }

    SUBCASE("Symmetric matrix") {
      MESSAGE("Test a 4x4 cost matrix that's symmetrical");
      int dim = 4;
      Eigen::MatrixXd costMatrix(dim, dim);
      costMatrix << 1.0, 2.0, 4.0, 16.0, 2.0, 3.0, 9.0, 81.0, 4.0, 9.0, 5.0,
          25.0, 16.0, 81.0, 25.0, 7.0;

      HungarianOptimizer<double> optimizer(costMatrix);
      const auto assignments = optimizer.minimize();
      const auto& rowsol = assignments.first;
      const auto& colsol = assignments.second;
      const auto totalCost =
          HungarianOptimizer<double>::getTotalCost(costMatrix, rowsol, colsol);
      HungarianOptimizer<double>::printMatrix(costMatrix, rowsol, colsol);

      // Check expected values
      auto expectedCost = 16.0d;
      CHECK_EQ(int((totalCost - expectedCost) * 100), 0);
      Eigen::VectorXi expRowsol(dim);
      expRowsol << 0, 1, 2, 3;
      Eigen::VectorXi expColsol1(dim);
      expColsol1 << 0, 1, 2, 3;
      Eigen::VectorXi expColsol2(dim);
      expColsol2 << 1, 0, 2, 3;
      CHECK_EQ(rowsol, expRowsol);
      CHECK_EQ((colsol == expColsol1 || colsol == expColsol2), true);
    }
  }

  TEST_CASE("Test 3x4 cost matrix") {
    SUBCASE("Optimal diagonal solution") {
      MESSAGE("Test a 3x4 cost matrix with an optimal diagonal solution");
      int rows = 3;
      int cols = 4;
      Eigen::MatrixXd costMatrix(rows, cols);
      costMatrix << 1.2, 3.23, 2.54, 4.94, 9.1, 2.22, 5.21, 3.23, 7.1, 4.3, 0.2,
          8.93;

      HungarianOptimizer<double> optimizer(costMatrix);
      const auto assignments = optimizer.minimize();
      const auto& rowsol = assignments.first;
      const auto& colsol = assignments.second;
      const auto totalCost =
          HungarianOptimizer<double>::getTotalCost(costMatrix, rowsol, colsol);
      HungarianOptimizer<double>::printMatrix(costMatrix, rowsol, colsol);

      // Check expected values
      auto expectedCost = 3.62;
      CHECK_EQ(int((totalCost - expectedCost) * 10000000000000000), 0);
      Eigen::VectorXi expRowsol(std::min(rows, cols));
      expRowsol << 0, 1, 2;
      Eigen::VectorXi expColsol(std::min(rows, cols));
      expColsol << 0, 1, 2;
      CHECK_EQ(rowsol, expRowsol);
      CHECK_EQ(colsol, expColsol);
    }

    SUBCASE("Optimal reversed diagonal solution") {
      MESSAGE(
          "Test a 3x4 cost matrix with an optimal reversed diagonal solution");
      int rows = 3;
      int cols = 4;
      Eigen::MatrixXd costMatrix(rows, cols);
      costMatrix << 4.94, 3.23, 2.54, 1.2, 9.1, 5.21, 2.22, 3.23, 7.1, 0.2, 4.3,
          8.93;

      HungarianOptimizer<double> optimizer(costMatrix);
      const auto assignments = optimizer.minimize();
      const auto& rowsol = assignments.first;
      const auto& colsol = assignments.second;
      const auto totalCost =
          HungarianOptimizer<double>::getTotalCost(costMatrix, rowsol, colsol);
      HungarianOptimizer<double>::printMatrix(costMatrix, rowsol, colsol);

      // Check expected values
      auto expectedCost = 3.62d;
      CHECK_EQ(int((totalCost - expectedCost) * 10000000000000000), 0);
      Eigen::VectorXi expRowsol(std::min(rows, cols));
      expRowsol << 0, 1, 2;
      Eigen::VectorXi expColsol(std::min(rows, cols));
      expColsol << 3, 2, 1;
      CHECK_EQ(rowsol, expRowsol);
      CHECK_EQ(colsol, expColsol);
    }

    SUBCASE("All decimal values") {
      MESSAGE("Test a 3x4 cost matrix with all decimal values");
      int rows = 3;
      int cols = 4;
      Eigen::MatrixXd costMatrix(rows, cols);
      costMatrix << 0.2, 0.23, 0.54, 0.2, 0.1, 0.2, 0.22, 0.23, 0.3, 0.2, 0.2,
          0.93;

      HungarianOptimizer<double> optimizer(costMatrix);
      const auto assignments = optimizer.minimize();
      const auto& rowsol = assignments.first;
      const auto& colsol = assignments.second;
      const auto totalCost =
          HungarianOptimizer<double>::getTotalCost(costMatrix, rowsol, colsol);
      HungarianOptimizer<double>::printMatrix(costMatrix, rowsol, colsol);

      // Check expected values
      auto expectedCost = 0.5d;
      CHECK_EQ(int((totalCost - expectedCost) * 100), 0);
      Eigen::VectorXi expRowsol(std::min(rows, cols));
      expRowsol << 0, 1, 2;
      Eigen::VectorXi expColsol(std::min(rows, cols));
      expColsol << 3, 0, 1;
      CHECK_EQ(rowsol, expRowsol);
      CHECK_EQ(colsol, expColsol);
    }

    SUBCASE("Symmetric matrix") {
      MESSAGE("Test a 3x4 cost matrix that's symmetrical");
      int rows = 3;
      int cols = 4;
      Eigen::MatrixXd costMatrix(rows, cols);
      costMatrix << 1.0, 2.0, 4.0, 16.0, 2.0, 3.0, 9.0, 81.0, 4.0, 9.0, 5.0,
          25.0;

      HungarianOptimizer<double> optimizer(costMatrix);
      const auto assignments = optimizer.minimize();
      const auto& rowsol = assignments.first;
      const auto& colsol = assignments.second;
      const auto totalCost =
          HungarianOptimizer<double>::getTotalCost(costMatrix, rowsol, colsol);
      HungarianOptimizer<double>::printMatrix(costMatrix, rowsol, colsol);

      // Check expected values
      auto expectedCost = 9.0d;
      CHECK_EQ(int((totalCost - expectedCost) * 100), 0);
      Eigen::VectorXi expRowsol(std::min(rows, cols));
      expRowsol << 0, 1, 2;
      Eigen::VectorXi expColsol1(std::min(rows, cols));
      expColsol1 << 0, 1, 2;
      Eigen::VectorXi expColsol2(std::min(rows, cols));
      expColsol2 << 1, 0, 2;
      CHECK_EQ(rowsol, expRowsol);
      CHECK_EQ((colsol == expColsol1 || colsol == expColsol2), true);
    }
  }

  TEST_CASE("Test 4x3 cost matrix") {
    SUBCASE("Optimal diagonal solution") {
      MESSAGE("Test a 4x3 cost matrix with an optimal diagonal solution");
      int rows = 4;
      int cols = 3;
      Eigen::MatrixXd costMatrix(rows, cols);
      costMatrix << 1.2, 3.23, 2.54, 9.1, 2.22, 5.21, 7.1, 4.3, 0.2, 6.34, 1.23,
          8.11;

      HungarianOptimizer<double> optimizer(costMatrix);
      const auto assignments = optimizer.minimize();
      const auto& rowsol = assignments.first;
      const auto& colsol = assignments.second;
      const auto totalCost =
          HungarianOptimizer<double>::getTotalCost(costMatrix, rowsol, colsol);
      HungarianOptimizer<double>::printMatrix(costMatrix, rowsol, colsol);

      // Check expected values
      auto expectedCost = 2.63d;
      CHECK_EQ(int((totalCost - expectedCost) * 10000000000000000), 0);
      Eigen::VectorXi expRowsol(std::min(rows, cols));
      expRowsol << 0, 2, 3;
      Eigen::VectorXi expColsol(std::min(rows, cols));
      expColsol << 0, 2, 1;
      CHECK_EQ(rowsol, expRowsol);
      CHECK_EQ(colsol, expColsol);
    }

    SUBCASE("Optimal reversed diagonal solution") {
      MESSAGE(
          "Test a 4x3 cost matrix with an optimal reversed diagonal solution");
      int rows = 4;
      int cols = 3;
      Eigen::MatrixXd costMatrix(rows, cols);
      costMatrix << 4.94, 3.23, 2.54, 9.1, 5.21, 2.22, 7.1, 0.2, 4.3, 1.94,
          1.23, 8.11;

      HungarianOptimizer<double> optimizer(costMatrix);
      const auto assignments = optimizer.minimize();
      const auto& rowsol = assignments.first;
      const auto& colsol = assignments.second;
      const auto totalCost =
          HungarianOptimizer<double>::getTotalCost(costMatrix, rowsol, colsol);
      HungarianOptimizer<double>::printMatrix(costMatrix, rowsol, colsol);

      // Check expected values
      auto expectedCost = 4.36d;
      CHECK_EQ(int((totalCost - expectedCost) * 10000000000000000), 0);
      Eigen::VectorXi expRowsol(std::min(rows, cols));
      expRowsol << 1, 2, 3;
      Eigen::VectorXi expColsol(std::min(rows, cols));
      expColsol << 2, 1, 0;
      CHECK_EQ(rowsol, expRowsol);
      CHECK_EQ(colsol, expColsol);
    }

    SUBCASE("All decimal values") {
      MESSAGE("Test a 4x3 cost matrix with all decimal values");
      int rows = 4;
      int cols = 3;
      Eigen::MatrixXd costMatrix(rows, cols);
      costMatrix << 0.2, 0.23, 0.54, 0.1, 0.2, 0.22, 0.3, 0.2, 0.2, 0.94, 0.23,
          0.11;

      HungarianOptimizer<double> optimizer(costMatrix);
      const auto assignments = optimizer.minimize();
      const auto& rowsol = assignments.first;
      const auto& colsol = assignments.second;
      const auto totalCost =
          HungarianOptimizer<double>::getTotalCost(costMatrix, rowsol, colsol);
      HungarianOptimizer<double>::printMatrix(costMatrix, rowsol, colsol);

      // Check expected values
      auto expectedCost = 0.41000000000000003d;
      CHECK_EQ(int((totalCost - expectedCost) * 100), 0);
      Eigen::VectorXi expRowsol(std::min(rows, cols));
      expRowsol << 1, 2, 3;
      Eigen::VectorXi expColsol(std::min(rows, cols));
      expColsol << 0, 1, 2;
      CHECK_EQ(rowsol, expRowsol);
      CHECK_EQ(colsol, expColsol);
    }

    SUBCASE("Symmetric matrix") {
      MESSAGE("Test a 4x3 cost matrix that's symmetrical");
      int rows = 4;
      int cols = 3;
      Eigen::MatrixXd costMatrix(rows, cols);
      costMatrix << 1.0, 2.0, 4.0, 2.0, 3.0, 9.0, 4.0, 9.0, 5.0, 16.0, 81.0,
          25.0;

      HungarianOptimizer<double> optimizer(costMatrix);
      const auto assignments = optimizer.minimize();
      const auto& rowsol = assignments.first;
      const auto& colsol = assignments.second;
      const auto totalCost =
          HungarianOptimizer<double>::getTotalCost(costMatrix, rowsol, colsol);
      HungarianOptimizer<double>::printMatrix(costMatrix, rowsol, colsol);

      // Check expected values
      auto expectedCost = 9.0d;
      CHECK_EQ(int((totalCost - expectedCost) * 100), 0);
      Eigen::VectorXi expRowsol(std::min(rows, cols));
      expRowsol << 0, 1, 2;
      Eigen::VectorXi expColsol1(std::min(rows, cols));
      expColsol1 << 0, 1, 2;
      Eigen::VectorXi expColsol2(std::min(rows, cols));
      expColsol2 << 1, 0, 2;
      CHECK_EQ(rowsol, expRowsol);
      CHECK_EQ((colsol == expColsol1 || colsol == expColsol2), true);
    }
  }

  TEST_CASE("Test Hungarian algorithm performance") {
    SUBCASE("Test performance with 100x100 cost matrix") {
      MESSAGE("Testing performance with a random 100x100 cost matrix");
      std::random_device rd;
      auto seed_data = std::array<double, std::mt19937::state_size>{};
      std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
      std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
      std::mt19937 generator(seq);
      std::uniform_int_distribution<> dis(0.0d, 1.0d);

      int dim = 100;
      Eigen::MatrixXd costMatrix(dim, dim);
      for (size_t i = 0; i < dim; ++i) costMatrix(i, i) = dis(generator);

      const auto start = std::chrono::high_resolution_clock::now();
      HungarianOptimizer<double> optimizer(costMatrix);
      const auto assignments = optimizer.minimize();
      const auto totalCost = optimizer.getTotalCost(
          costMatrix, assignments.first, assignments.second);
      const auto totalEnd = std::chrono::high_resolution_clock::now();
      const auto totalTook =
          std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd -
                                                                start)
              .count();
      std::cout << "Finding optimal assignments took: " << totalTook << "ms"
                << std::endl;
    }

    SUBCASE("Test performance with 1000x1000 cost matrix") {
      MESSAGE("Testing performance with a random 1000x1000 cost matrix");
      std::random_device rd;
      auto seed_data = std::array<double, std::mt19937::state_size>{};
      std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
      std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
      std::mt19937 generator(seq);
      std::uniform_int_distribution<> dis(0.0d, 1.0d);

      int dim = 1000;
      Eigen::MatrixXd costMatrix(dim, dim);
      for (size_t i = 0; i < dim; ++i) costMatrix(i, i) = dis(generator);

      const auto start = std::chrono::high_resolution_clock::now();
      HungarianOptimizer<double> optimizer(costMatrix);
      const auto assignments = optimizer.minimize();
      const auto totalCost = optimizer.getTotalCost(
          costMatrix, assignments.first, assignments.second);
      const auto totalEnd = std::chrono::high_resolution_clock::now();
      const auto totalTook =
          std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd -
                                                                start)
              .count();
      std::cout << "Finding optimal assignments took: " << totalTook << "ms"
                << std::endl;
    }

    SUBCASE("Test performance with 10000x10000 cost matrix") {
      MESSAGE("Testing performance with a random 10000x10000 cost matrix");
      std::random_device rd;
      auto seed_data = std::array<double, std::mt19937::state_size>{};
      std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
      std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
      std::mt19937 generator(seq);
      std::uniform_int_distribution<> dis(0.0d, 1.0d);

      int dim = 10000;
      Eigen::MatrixXd costMatrix(dim, dim);
      for (size_t i = 0; i < dim; ++i) costMatrix(i, i) = dis(generator);

      const auto start = std::chrono::high_resolution_clock::now();
      HungarianOptimizer<double> optimizer(costMatrix);
      const auto assignments = optimizer.minimize();
      const auto totalCost = optimizer.getTotalCost(
          costMatrix, assignments.first, assignments.second);
      const auto totalEnd = std::chrono::high_resolution_clock::now();
      const auto totalTook =
          std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd -
                                                                start)
              .count();
      std::cout << "Finding optimal assignments took: " << totalTook << "ms"
                << std::endl;
    }
  }
}
