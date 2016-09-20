import pandas as pd
import numpy as np
from pyliftover import LiftOver
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit

from NucFrames.monte_carlo_utils import random_sample

def read_loop_frame(loop_file):
  """
  Utility to read Rao et al loops file.
  """
  df = pd.read_csv(loop_file,
                   sep="\t",
                   dtype={'chr1': str,
                          'x1': np.int32,
                          'x2': np.int32,
                          'chr2': str,
                          'y1': np.int32,
                          'y2': np.int32})

  df["x"] = df[["x1", "x2"]].mean(axis=1)
  df["y"] = df[["y1", "y2"]].mean(axis=1)

  lo = LiftOver('mm9', 'mm10')
  df["x"] = df.apply(lambda x: row_liftover(x, "chr1", "x", lo), axis=1)
  df["y"] = df.apply(lambda x: row_liftover(x, "chr2", "y", lo), axis=1)
  df["chr1"] = df.apply(lambda x: x["chr1"][3:], axis=1)
  df["chr2"] = df.apply(lambda x: x["chr2"][3:], axis=1)

  df = df[["chr1", "x", "chr2", "y"]]

  df.rename(columns={"x": "a", "y": "b"}, inplace=True)

  return (df)


def row_liftover(row, chr_col, col, lo):
  """
    Utility to convert bps in rows of a dataframe.
    Probably should return nan if exception.
    Assumes that chr_col is of the form chr2, not just 2.
    """
  bp_arr = lo.convert_coordinate(row[chr_col], row[col])
  bp = int(bp_arr[0][1])
  return (bp)


##############################################################################
# Monte Carlo stuff
##############################################################################
def monte_carlo(dists_count, seps, loop_total, idx_a, idx_b, B=100000, plot=False, n_cells=None):
  """
  Only on cis (obviously), assume square matrix.
  """
  max_idx = dists_count.shape[0] - 1

  if n_cells is None:
    n_cells = np.max(dists_count)

  potential_max = len(seps) * n_cells

  sample_starts, sample_ends = random_sample(seps, max_idx, B)
  #sample_starts, sample_ends = sanity_check(idx_a, idx_b, max_idx, B)

  sample_sums = mk_sum(dists_count, sample_starts, sample_ends, plot=plot)
  truth_count = np.sum(sample_sums >= loop_total)

  if plot:
    print("Max from background: {}, loop value: {}, theoretical max: {}".format(
            np.max(sample_sums), loop_total, potential_max))

    sns.distplot(sample_sums, kde=False, norm_hist=False)
    plt.axvline(loop_total, label="loop total")
    plt.legend()
    plt.show()

  pval = (1 + truth_count) / (1 + B)
  return(truth_count, pval)

def mk_sum(dists_count, sample_starts, sample_ends, plot=False):
  sample_values = np.zeros_like(sample_starts)
  _mk_sum(dists_count, sample_values, sample_starts, sample_ends)
  highest = np.sum(np.max(sample_values, axis=0))
  lowest = np.sum(np.min(sample_values, axis=0))
  if plot:
    print("Highest possible sum from sampling: {}".format(highest))
    print("Lowest possible sum from sampling: {}".format(lowest))
    plt.axvline(highest, color='r', label="Highest possible from sampling")
    plt.axvline(lowest, color='r', label="Lowest possible from sampling")

  return(np.sum(sample_values, axis=1))

@jit(nopython=True, nogil=True)
def _mk_sum(dists_count, sample_values, sample_starts, sample_ends):
  for sample_idx in range(sample_starts.shape[0]):
    for sep_idx in range(sample_starts.shape[1]):
      idx_a = sample_starts[sample_idx, sep_idx]
      idx_b = sample_ends[sample_idx, sep_idx]
      sample_values[sample_idx, sep_idx] = dists_count[idx_a, idx_b]

def sanity_check(idx_a, idx_b, max_idx, B):
  """
  Start with the known loop configuration, randomly shuffle positions from there.
  """
  samples_a = []
  samples_b = []

  for i in range(idx_a.shape[0]):
    a = idx_a[i]
    b = idx_b[i]

    # loop_changes = np.random.randint(-a, max_idx - b, B)
    val = np.min([a, max_idx - b - 1])

    loop_changes = np.random.binomial(n=2 * val, p=0.5, size=B)
    loop_changes = loop_changes - val

    loop_changes[(loop_changes + b) >= max_idx] = max_idx - 1
    loop_changes[loop_changes < 0] = 0

    starts = a + loop_changes.copy()
    ends = b + loop_changes

    samples_a.append(starts)
    samples_b.append(ends)

  samples_a = np.array(samples_a, dtype=np.int32).T
  samples_b = np.array(samples_b, dtype=np.int32).T
  return(samples_a, samples_b)

def circular_permute(idx_a, idx_b, max_idx):
  samples_a = []
  samples_b = []

  seps = np.abs(idx_a - idx_b)

  for i in range(100 * max_idx):
    idx_a = idx_a + 1
    idx_b = idx_b + 1

    if np.sum(idx_b > max_idx) > 0:
        pass
    else:
      overlap = np.argwhere(idx_a == max_idx)

      idx_a[overlap] = 0
      idx_b[overlap] = seps[overlap]

      samples_a.append(idx_a)
      samples_b.append(idx_b)
  return (np.array(samples_a), np.array(samples_b))


def loop_monte_carlo(idx_a, idx_b, background, B=10000):
  """
  Assess how unlikely a single loop is, across all cells.
  Background is the summed along cells pseudocontact matrix for that chromosome.
  """

  sep = np.abs(idx_b - idx_a)
  random_starts = np.random.randint(0, background.shape[0] - sep, B)
  random_ends = random_starts + sep

  target_total = background[idx_a, idx_b]

  random_totals = background[random_starts, random_ends]

  truth_count = np.sum(target_total <= random_totals)

  p = (1 + truth_count) / (1 + B)

  return(p)


if __name__ == "__main__":
  loop_file = "/home/lpa24/dev/cam/data/rao_et_al_data/GSE63525_CH12-LX_HiCCUPS_looplist_with_motifs.txt"
  df = read_loop_frame(loop_file)
  print(df)
