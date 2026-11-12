# Parallel Word Counting in a Text Corpus with MPI

## 1. Problem Description
Count how many times each word in `dataset/consulta.txt` appears
in a corpus of 3000 files `file_XXXX.txt`, and report the
top 10. Three implementations are compared: a sequential baseline, an MPI
version with static file distribution (v1), and an MPI version with improved
load balance (v2).

## 2. Environment and Execution Instructions

**Hardware/software:** HP 14-dq2504la, Intel Core i3-1115G4 (2 cores / 4
threads), 16 GB RAM, Windows 11, Docker Desktop (WSL2), image
`augustosalazar/slim-mpi:2`. Project is located in OneDrive.

### a. Experimental plan
1. Generate the dataset with `generator.py` (seed 42, reproducible).
2. Run the sequential baseline 5 times; the first run is a cold-cache
   warm-up and is excluded from T_seq.
3. Determine p with `Detectar_procesos_mpi.py` (result: p = 4).
4. Run MPI v1 at p = 4, at least 3 runs; record total and per-process times.
5. Run MPI v2 under the same conditions.
6. Compute T_p, S_p = T_seq / T_p, E_p = S_p / p and LI = max(t_i)/mean(t_i).

### b. Sequential baseline


### c. MPI version 1


### d. Test procedure


## 3. Experimental Plan Execution

### a. Sequential baseline timing
| Run | Time (s) | Note |
|---|---|---|
| 1 | 110.155 | First time, excluded, reason: cache |
| 2 | 35.933 | |
| 3 | 38.959 | |
| 4 | 30.330 | |
| 5 | 32.326 | |

T_seq = 34.39 s (mean of runs 2-5, std is approximately 3.8 s).

Reference top 10 (3000 files, 44,951,458 tokens, 3,631,778 occurrences):
a: 785774, para: 392156, sus: 228913, otros: 105530, ante: 99832,
unos: 88794, otra: 83901, vosotros: 61617, mios: 58420, tuya: 56635.

### Choice of p
`Detectar_procesos_mpi.py` (prime counting, 3 measurements per N):

| N | Median (s) |
|---|---|
| 1 | 6.408 |
| 2 | 5.793 |
| 3 | 7.018 |
| 4 | 5.050 |

Recommended N = 4, so p = 4.

### b. MPI v1 timing results (p = 4)

| Run | Total (s) | Setup (s) | Max local (s) | LI |
|---|---|---|---|---|
| 1 | 29.492 | 7.87 | 21.623 | 1.0315 |
| 2 | 23.107 | 7.04 | 16.087 | 1.0438 |
| 3 | 23.755 | 5.87 | 17.888 | 1.0512 |
| 4 | 22.607 | 6.67 | 15.932 | 1.0593 |
| 5 | 22.805 | 5.51 | 17.293 | 1.0475 |

T_p = 24.35 s (std 2.91 s), S_p = 34.39 / 24.35 = 1.41, E_p = 1.41 / 4 = 0.35.
Run 1 is a slow outlier. Without it: T_p = 23.07 s, S_p = 1.49, E_p = 0.37.
All runs matched the baseline output (`MATCHES_BASELINE=True`).

### c. Load imbalance evidence (v1)

Mean over 5 runs:

| Rank | Files | Tokens | Mean local time (s) |
|---|---|---|---|
| 0 | 750 | 10,803,859 | 16.57 |
| 1 | 750 | 12,156,824 | 17.77 |
| 2 | 750 | 10,816,979 | 16.49 |
| 3 | 750 | 11,173,796 | 17.13 |

Mean LI = 1.047 (range 1.032-1.059), mean relative spread = 0.083.
Rank 1, which received the most tokens, was the slowest in all 5 runs.
Perfect balance would reduce the slowest process by ~0.78 s (about 3 % of T_p).
Setup time (reading the query, listing files, bcast, scatter) was ~6.6 s on
average (27 % of T_p) and is spent by rank 0 while the other ranks wait.


### d. MPI v2 results


## 4. Analysis
### a. Parallel performance


### b. Load balance


### c. Limitations



## 5. Conclusions

