"""Finds a matrix have the largest determinant in `n*n` dimensions.
On every iteration, improve priority_v1 over the priority_vX methods from previous iterations.
Make only small changes.
Try to make the code short.
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
  """ Generate a list of indices for an n*n matrix filled in diagonal order """
  indices = []
  for k in range(2 * n - 1):
      for i in range(n):
          j = k - i
          if 0 <= j < n:
              indices.append((i, j))
  return indices

def fill_diagonal_matrix(lst, n):
  """ Fill the list lst into an n*n matrix, in diagonal order. """
  if len(lst) != n * n:
      raise ValueError("lst must have n*n elements.")
  
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
  """Returns the priority with which we want to add `element` to the matrix.
  el is a number of values 1-n*n.
  """
  return 0.0
