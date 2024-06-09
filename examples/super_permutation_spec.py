"""Finds short superpermutation
On every iteration, improve priority_v1 over the priority_vX methods from previous iterations.
Make only small changes.
Try to make the code short.
"""
import itertools

import funsearch


@funsearch.run
def evaluate(n: int) -> int:
  """
  Returns the length of the super permutation for a given n.

  :param n: Number of elements to generate the super permutation for.
  :return: Length of the super permutation.
  """
  super_permutation = solve(n)
  # https://oeis.org/A180632
  best_known_score = {1: 1, 2: 3, 3: 9, 4: 33, 5: 153, 6: 872, 7: 5908, 8: 46205, 9: 408966}
  return best_known_score[n]-len(super_permutation)


def solve(n: int) -> str:
  """Returns a super permutation for given n."""
  permutations = itertools.permutations(range(1, n + 1))
  perm_strs = [''.join(map(str, p)) for p in permutations]

  super_perm = perm_strs[0]
  used = {super_perm: True}

  while len(used) < len(perm_strs):
    max_priority, next_perm = -1, None
    for perm in perm_strs:
      if perm not in used:
        priority_score = priority(super_perm, perm, n)
        if priority_score > max_priority:
          max_priority, next_perm = priority_score, perm
    overlap = next((i for i in range(len(next_perm), 0, -1) if super_perm.endswith(next_perm[:i])), 0)
    super_perm += next_perm[overlap:]
    used[next_perm] = True
  return super_perm

@funsearch.evolve
def priority(super_perm, next_perm, n) -> float:
  """Returns the priority of the next_perm. the longer overlap, the higher priority it is"""
  overlap = len(super_perm) - 1
  while overlap > 0 and super_perm[-overlap:] != next_perm[:overlap]:
    overlap -= 1
  return overlap / len(next_perm)

print(evaluate(7)) 