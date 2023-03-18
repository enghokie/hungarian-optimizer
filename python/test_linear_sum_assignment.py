import numpy as np
import datetime
from scipy.optimize import linear_sum_assignment

print('Test a 4x4 cost matrix with an optimal diagonal solution')
cost = np.array([[1.2, 3.23, 2.54, 4.94], [9.1, 2.22, 5.21, 3.23], [7.1, 4.3, 0.2, 8.93], [6.34, 1.23, 8.11, 1.94]])
row_ind, col_ind = linear_sum_assignment(cost)
total_cost = cost[row_ind, col_ind].sum()
print(f'total_cost: {total_cost}, row_ind:\n{row_ind}\ncol_ind:\n{col_ind}')

assert(total_cost == 5.5600000000000005)
assert(np.array_equal(row_ind,np.array([0, 1, 2, 3])))
assert(np.array_equal(col_ind, np.array([0, 1, 2, 3])))

print('\n====\n')

print('Test a 4x4 cost matrix with an optimal reversed diagonal solution')
cost = np.array([[4.94, 3.23, 2.54, 1.2], [9.1, 5.21, 2.22, 3.23], [7.1, 0.2, 4.3, 8.93], [1.94, 1.23, 8.11, 6.34]])
row_ind, col_ind = linear_sum_assignment(cost)
total_cost = cost[row_ind, col_ind].sum()
print(f'total_cost: {total_cost}, row_ind:\n{row_ind}\ncol_ind:\n{col_ind}')

assert(total_cost == 5.5600000000000005)
assert(np.array_equal(row_ind, np.array([0, 1, 2, 3])))
assert(np.array_equal(col_ind, np.array([3, 2, 1, 0])))

print('\n====\n')

print('Test a 4x4 cost matrix with all decimal values')
cost = np.array([[0.2, 0.23, 0.54, 0.2], [0.1, 0.2, 0.22, 0.23], [0.3, 0.2, 0.2, 0.93], [0.94, 0.23, 0.11, 0.94]])
row_ind, col_ind = linear_sum_assignment(cost)
total_cost = cost[row_ind, col_ind].sum()
print(f'total_cost: {total_cost}, row_ind:\n{row_ind}\ncol_ind:\n{col_ind}')

assert(total_cost == 0.61)
assert(np.array_equal(row_ind, np.array([0, 1, 2, 3])))
assert(np.array_equal(col_ind, np.array([3, 0, 1, 2])))

print('\n====\n')

print('Test a 4x4 cost matrix that is symmetrical')
cost = np.array([[1.0, 2.0, 4.0, 16.0], [2.0, 3.0, 9.0, 81.0], [4.0, 9.0, 5.0, 25.0], [16.0, 81.0, 25.0, 7.0]])
row_ind, col_ind = linear_sum_assignment(cost)
total_cost = cost[row_ind, col_ind].sum()
print(f'total_cost: {total_cost}, row_ind:\n{row_ind}\ncol_ind:\n{col_ind}')

assert(total_cost == 16.0)
assert(np.array_equal(row_ind, np.array([0, 1, 2, 3])))
assert(np.array_equal(col_ind, np.array([0, 1, 2, 3])) or np.array_equal(col_ind, np.array([1, 0, 2, 3])))

print('\n====\n')

print('Test a 3x4 cost matrix with an optimal diagonal solution')
cost = np.array([[1.2, 3.23, 2.54, 4.94], [9.1, 2.22, 5.21, 3.23], [7.1, 4.3, 0.2, 8.93]])
row_ind, col_ind = linear_sum_assignment(cost)
total_cost = cost[row_ind, col_ind].sum()
print(f'total_cost: {total_cost}, row_ind:\n{row_ind}\ncol_ind:\n{col_ind}')

assert(total_cost == 3.62)
assert(np.array_equal(row_ind,np.array([0, 1, 2])))
assert(np.array_equal(col_ind, np.array([0, 1, 2])))

print('\n====\n')

print('Test a 3x4 cost matrix with an optimal reversed diagonal solution')
cost = np.array([[4.94, 3.23, 2.54, 1.2], [9.1, 5.21, 2.22, 3.23], [7.1, 0.2, 4.3, 8.93]])
row_ind, col_ind = linear_sum_assignment(cost)
total_cost = cost[row_ind, col_ind].sum()
print(f'total_cost: {total_cost}, row_ind:\n{row_ind}\ncol_ind:\n{col_ind}')

assert(total_cost == 3.62)
assert(np.array_equal(row_ind, np.array([0, 1, 2])))
assert(np.array_equal(col_ind, np.array([3, 2, 1])))

print('\n====\n')

print('Test a 3x4 cost matrix with all decimal values')
cost = np.array([[0.2, 0.23, 0.54, 0.2], [0.1, 0.2, 0.22, 0.23], [0.3, 0.2, 0.2, 0.93]])
row_ind, col_ind = linear_sum_assignment(cost)
total_cost = cost[row_ind, col_ind].sum()
print(f'total_cost: {total_cost}, row_ind:\n{row_ind}\ncol_ind:\n{col_ind}')

assert(total_cost == 0.5)
assert(np.array_equal(row_ind, np.array([0, 1, 2])))
assert(np.array_equal(col_ind, np.array([3, 0, 1])))

print('\n====\n')

print('Test a 3x4 cost matrix that is symmetrical')
cost = np.array([[1.0, 2.0, 4.0, 16.0], [2.0, 3.0, 9.0, 81.0], [4.0, 9.0, 5.0, 25.0]])
row_ind, col_ind = linear_sum_assignment(cost)
total_cost = cost[row_ind, col_ind].sum()
print(f'total_cost: {total_cost}, row_ind:\n{row_ind}\ncol_ind:\n{col_ind}')

assert(total_cost == 9.0)
assert(np.array_equal(row_ind, np.array([0, 1, 2])))
assert(np.array_equal(col_ind, np.array([0, 1, 2])) or np.array_equal(col_ind, np.array([1, 0, 2])))

print('\n====\n')

print('Test a 4x3 cost matrix with an optimal diagonal solution')
cost = np.array([[1.2, 3.23, 2.54], [9.1, 2.22, 5.21], [7.1, 4.3, 0.2], [6.34, 1.23, 8.11]])
row_ind, col_ind = linear_sum_assignment(cost)
total_cost = cost[row_ind, col_ind].sum()
print(f'total_cost: {total_cost}, row_ind:\n{row_ind}\ncol_ind:\n{col_ind}')

assert(total_cost == 2.63)
assert(np.array_equal(row_ind,np.array([0, 2, 3])))
assert(np.array_equal(col_ind, np.array([0, 2, 1])))

print('\n====\n')

print('Test a 4x3 cost matrix with an optimal reversed diagonal solution')
cost = np.array([[4.94, 3.23, 2.54], [9.1, 5.21, 2.22], [7.1, 0.2, 4.3], [1.94, 1.23, 8.11]])
row_ind, col_ind = linear_sum_assignment(cost)
total_cost = cost[row_ind, col_ind].sum()
print(f'total_cost: {total_cost}, row_ind:\n{row_ind}\ncol_ind:\n{col_ind}')

assert(total_cost == 4.36)
assert(np.array_equal(row_ind, np.array([1, 2, 3])))
assert(np.array_equal(col_ind, np.array([2, 1, 0])))

print('\n====\n')

print('Test a 4x3 cost matrix with all decimal values')
cost = np.array([[0.2, 0.23, 0.54], [0.1, 0.2, 0.22], [0.3, 0.2, 0.2], [0.94, 0.23, 0.11]])
row_ind, col_ind = linear_sum_assignment(cost)
total_cost = cost[row_ind, col_ind].sum()
print(f'total_cost: {total_cost}, row_ind:\n{row_ind}\ncol_ind:\n{col_ind}')

assert(total_cost == 0.41000000000000003)
assert(np.array_equal(row_ind, np.array([1, 2, 3])))
assert(np.array_equal(col_ind, np.array([0, 1, 2])))

print('\n====\n')

print('Test a 4x3 cost matrix that is symmetrical')
cost = np.array([[1.0, 2.0, 4.0], [2.0, 3.0, 9.0], [4.0, 9.0, 5.0], [16.0, 81.0, 25.0]])
row_ind, col_ind = linear_sum_assignment(cost)
total_cost = cost[row_ind, col_ind].sum()
print(f'total_cost: {total_cost}, row_ind:\n{row_ind}\ncol_ind:\n{col_ind}')

assert(total_cost == 9.0)
assert(np.array_equal(row_ind, np.array([0, 1, 2])))
assert(np.array_equal(col_ind, np.array([0, 1, 2])) or np.array_equal(col_ind, np.array([1, 0, 2])))

print('\n====\n')

print('Test performance with a random 100x100 cost matrix')
cost = np.random.random_sample((100, 100))
start_time = datetime.datetime.now()
row_ind, col_ind = linear_sum_assignment(cost)
total_cost = cost[row_ind, col_ind].sum()
elapsed = datetime.datetime.now() - start_time
print(f'{elapsed.total_seconds() * 1000}ms')

print('\n====\n')

print('Test performance with a random 1000x1000 cost matrix')
cost = np.random.random_sample((1000, 1000))
start_time = datetime.datetime.now()
row_ind, col_ind = linear_sum_assignment(cost)
total_cost = cost[row_ind, col_ind].sum()
elapsed = datetime.datetime.now() - start_time
print(f'{elapsed.total_seconds() * 1000}ms')

print('\n====\n')

print('Test performance with a random 10000x10000 cost matrix')
cost = np.random.random_sample((10000, 10000))
start_time = datetime.datetime.now()
row_ind, col_ind = linear_sum_assignment(cost)
total_cost = cost[row_ind, col_ind].sum()
elapsed = datetime.datetime.now() - start_time
print(f'{elapsed.total_seconds() * 1000}ms')
