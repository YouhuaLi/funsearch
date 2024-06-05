"""Finds a matrix have largest determinant in `n*n` dimensions.
"""
import itertools

import numpy as np

import funsearch

import numpy as np


@funsearch.run
def evaluate(n: int) -> int:
  """Returns thedeterminant n by n matrix.
     if there is duplicate cells return -inf
  """
  mtr = solve(n)
  #print(mtr)
  return abs(round(np.linalg.det(mtr)))

def diagonal_indices(n: int) -> list:
  """ 生成n*n矩阵按对角线顺序填充的索引列表 """
  indices = []
  for k in range(2 * n - 1):
      for i in range(n):
          j = k - i
          if 0 <= j < n:
              indices.append((i, j))
  return indices

def fill_diagonal_matrix(lst, n):
  """ 将列表lst填充到一个n*n矩阵中，按对角线顺序 """
  if len(lst) != n * n:
      raise ValueError("列表长度不符合 n*n 的要求")
  
  # 创建一个 n*n 的零数组
  result = np.zeros((n, n), dtype=int)

  # 获取对角线顺序的索引
  indices = diagonal_indices(n)

  # 填充数组
  for index, value in zip(indices, lst):
      result[index] = value

  return result

def solve(n: int) -> np.ndarray:
  """Returns a matrx in `n*n` dimensions."""
  all_vectors = np.array(list(range(1, n*n+1)), dtype=np.int32)

  # Precompute all priorities.
  priorities = np.array([priority(vector, n) for vector in all_vectors])

  # Build `matrix_array` greedily, using priorities for prioritization.
  matrix_array = []
  while np.any(priorities != -np.inf):
    # Add a vector with maximum priority to `matrix_array`, and set priorities of
    # invalidated vectors to `-inf`, so that they never get selected.
    max_index = np.argmax(priorities)
    vector = all_vectors[None, max_index]  # [1, n]
    priorities[max_index] = -np.inf

    matrix_array.append(vector.item())
  
  return fill_diagonal_matrix(matrix_array, n)
  # return np.array(matrix_array).reshape(n,n)


@funsearch.evolve
def priority(el: int, n: int) -> float:
  """Improved version of `priority`."""
  # The determinant of a matrix with only one element is the element itself.
  if n == 1:
    return el

  # The determinant of a diagonal matrix is the product of its elements.
  # So, ideally, if we have some knowledge about the elements, we want them to be as largest as possible, to have a large determinant.
  if el > n:
    return n / el + np.random.uniform(0, 1) * 0.9
  # If no such knowledge exists, we assign a random priority, since any permutation could potentially lead to a large determinant.
  else:
    return np.random.uniform(0, 1) * 0.9
