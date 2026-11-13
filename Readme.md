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
7. Check that every run reproduces the baseline output.

### b. Sequential baseline

`baseline_secuencial.py` (provided, unchanged) loads the query words into a
set and iterates over every `file_*.txt` in `dataset/`. For each line it
splits the text into tokens, lowercases them and, for every token that is in
the query set, increments a `Counter`. It prints the elapsed time (measured
with `time.perf_counter` around the counting), the top 10 words, and writes
`dataset/baseline_results.csv`.

```powershell
docker run --rm -v "${pwd}:/app" augustosalazar/slim-mpi:2 python /app/baseline_secuencial.py
```
### c. MPI version 1

`mpi1.py` follows the steps of the assignment:

1. rank 0 reads `consulta.txt`;
2. rank 0 broadcasts the query words (`comm.bcast`);
3. rank 0 lists the `file_*.txt` files (sorted);
4. the files are split into `p` contiguous blocks whose sizes differ by at
   most one file, and distributed with `comm.scatter`. This balances the
   number of files, not the amount of work;
5. each process counts the query words in its own files (same counting logic
   as the baseline);
6. partial results are collected in rank 0 with `comm.gather`;
7. rank 0 merges them, prints the top 10 and compares the result with
   `dataset/baseline_results.csv`.

Timing: the clock starts after a `Barrier` and stops in rank 0 after the merge,
so it includes everything except process launch. Each process also reports its
number of assigned files, tokens, `setup_time` (everything before counting
starts) and `local_time` (counting only). The program prints `LOAD_IMBALANCE_LI`,
`RELATIVE_SPREAD`, `MATCHES_BASELINE` and `EXECUTION_TIME=<seconds>`.

### d. Test procedure

```powershell
# 1. Dataset
docker run --rm -v "${pwd}:/app" augustosalazar/slim-mpi:2 python /app/generator.py

# 2. Baseline (5 runs; the first one is the warm-up)
docker run --rm -v "${pwd}:/app" augustosalazar/slim-mpi:2 python /app/baseline_secuencial.py

# 3. Choice of p
docker run --rm -v "${pwd}:/app" augustosalazar/slim-mpi:2 python /app/Detectar_procesos_mpi.py

# 4. MPI v1 and v2, p = 4, 5 runs each (logs in logs/mpi1 and logs/mpi2)
mkdir logs\mpi1 -Force
1..5 | ForEach-Object { docker run --rm -v "${pwd}:/app" augustosalazar/slim-mpi:2 mpirun --allow-run-as-root --oversubscribe -np 4 python /app/mpi1.py | Tee-Object -FilePath "logs\mpi1\mpi1_run$_.log" }
mkdir logs\mpi2 -Force
1..5 | ForEach-Object { docker run --rm -v "${pwd}:/app" augustosalazar/slim-mpi:2 mpirun --allow-run-as-root --oversubscribe -np 4 python /app/mpi2.py | Tee-Object -FilePath "logs\mpi2\mpi2_run$_.log" }
```

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


### d. MPI version 2: how it corrects the imbalance, and results

**Design.** Version 2 changes two things:
1. *Size-aware assignment (LPT).* Files are sorted by size (bytes) and each
   one is given, largest first, to the process with the fewest assigned bytes.
   Every process computes the same deterministic assignment, so no `scatter`
   is needed.
2. *Shorter sequential setup.* Rank 0 only lists file names (one `listdir`,
   no per-file `isfile`). The `getsize` calls are split among all processes
   (750 files each) and shared with `allgather`.

**Timing results (p = 4).**

| Run | Total (s) | Setup (s) | Max local (s) | LI |
|---|---|---|---|---|
| 1 | 22.158 | 1.38 | 20.765 | 1.0074 |
| 2 | 21.320 | 1.59 | 19.726 | 1.0158 |
| 3 | 19.587 | 1.88 | 17.709 | 1.0083 |
| 4 | 18.155 | 1.54 | 16.620 | 1.0076 |
| 5 | 18.269 | 1.33 | 16.938 | 1.0117 |

T_p = 19.90 s (std 1.80 s), S_p = 34.39 / 19.90 = 1.73, E_p = 0.43,
mean LI = 1.010. All runs matched the baseline output. A separate test run
(17.62 s) was done before the 5 reported runs and is not included.

**Comparison.**

| Metric | v1 | v2 |
|---|---|---|
| T_p (s) | 24.35 | 19.90 |
| Speedup | 1.41 | 1.73 |
| Efficiency | 0.35 | 0.43 |
| LI | 1.047 | 1.010 |
| Mean setup (s) | 6.59 | 1.54 |
| Mean of max local time (s) | 17.77 | 18.35 |

Bytes per process are equal to within 50 bytes (about 68.76 MB each) and
tokens to within 0.02 % (about 11.24 M each).
The 4.45 s reduction in T_p is explained by the 5.05 s reduction in setup.
The balance improvement could save at most about 0.6 s, which is below the
run-to-run noise, and the counting phase was actually 0.58 s slower on average in
v2 for reasons not isolated here.

## 4. Analysis
### a. Parallel performance


### b. Load balance


### c. Limitations



## 5. Conclusions

