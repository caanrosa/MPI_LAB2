#  MPI version 2: size-aware static distribution (LPT) with a parallel size scan.

#  Improvements over version 1:
#   * Load balance: files are assigned by SIZE (bytes), not by count. The
#     Longest Processing Time first (LPT) heuristic gives each file, from the
#     largest to the smallest, to the process with the least assigned bytes.
#   * Shorter sequential setup: rank 0 only lists file names (no per-file
#     isfile() call). The getsize() calls, which touch the file system once per
#     file, are split among all processes and shared with allgather.
#   * No scatter: every process computes the same deterministic LPT assignment
#     from the same sizes, and simply picks its own files.

# Flow:
#   1. rank 0 reads consulta.txt and lists the file_*.txt names
#   2. rank 0 broadcasts the query words and the file names (comm.bcast)
#   3. each process measures the size of 1/p of the files (strided slice)
#   4. sizes are shared with comm.allgather
#   5. every process computes the LPT assignment and takes its own files
#   6. each process counts the query words in its files
#   7. partial results are gathered in rank 0 (comm.gather), merged and printed
#

import heapq
import os
from collections import Counter

from mpi4py import MPI


CONSULTA_NAME = "consulta.txt"
CASE_SENSITIVE = False
TOP_N = 10


def load_query(consulta_path):
    # Read consulta.txt and return the list of target words.
    if not os.path.isfile(consulta_path):
        raise FileNotFoundError(f"Query file not found: {consulta_path}")
    with open(consulta_path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    if not CASE_SENSITIVE:
        words = [w.lower() for w in words]
    return list(dict.fromkeys(words))


def list_corpus_names(dataset_dir):
    # Return the sorted file_*.txt names (a single listdir, no per-file stat).
    return sorted(
        n for n in os.listdir(dataset_dir)
        if n.startswith("file_") and n.endswith(".txt")
    )


def assign_lpt(sizes, parts):
    
    # LPT assignment. sizes[i] is the size of file i.
    # Returns (assignment, loads): assignment[r] = sorted list of file indices
    # for process r, loads[r] = total bytes assigned to process r.
    # Deterministic: ties are broken by index and by rank, so every process
    # computing it from the same sizes gets the same result.
    
    order = sorted(range(len(sizes)), key=lambda i: (-sizes[i], i))
    heap = [(0, r) for r in range(parts)]  # (assigned bytes, rank)
    heapq.heapify(heap)
    assignment = [[] for _ in range(parts)]
    loads = [0] * parts
    for i in order:
        load, r = heapq.heappop(heap)
        assignment[r].append(i)
        loads[r] = load + sizes[i]
        heapq.heappush(heap, (loads[r], r))
    for r in range(parts):
        assignment[r].sort()
    return assignment, loads


def count_in_files(paths, target_set):
    
    # Count occurrences of target words in the given files.
    
    freq = Counter()
    tokens = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                words = line.split()
                if not CASE_SENSITIVE:
                    words = [w.lower() for w in words]
                tokens += len(words)
                for w in words:
                    if w in target_set:
                        freq[w] += 1
    return freq, tokens


def compare_with_baseline(freq_global, baseline_csv):
    
    # Return True/False if the result equals baseline_results.csv, None if absent.
    
    if not os.path.isfile(baseline_csv):
        return None
    expected = {}
    with open(baseline_csv, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            word, count = line.strip().rsplit(",", 1)
            expected[word] = int(count)
    return expected == dict(freq_global)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset")

    # Start the global clock with all processes synchronized.
    comm.Barrier()
    t_start = MPI.Wtime()

    # Step 1: rank 0 reads the query and lists the file names.
    payload = None
    if rank == 0:
        payload = (
            load_query(os.path.join(dataset_dir, CONSULTA_NAME)),
            list_corpus_names(dataset_dir),
        )

    # Step 2: broadcast query words and file names.
    query_words, names = comm.bcast(payload, root=0)
    target_set = set(query_words)

    # Step 3: each process measures a strided 1/p slice of the files.
    my_slice = [
        (i, os.path.getsize(os.path.join(dataset_dir, names[i])))
        for i in range(rank, len(names), size)
    ]

    # Step 4: share the sizes with everyone.
    sizes = [0] * len(names)
    for part in comm.allgather(my_slice):
        for i, s in part:
            sizes[i] = s

    # Step 5: identical LPT assignment on every process; take own files.
    assignment, loads = assign_lpt(sizes, size)
    my_indices = assignment[rank]
    my_files = [os.path.join(dataset_dir, names[i]) for i in my_indices]
    assigned_bytes = loads[rank]

    # Step 6: local counting (timed per process).
    # setup_time = everything before counting starts (read query, list names,
    # bcast, size scan, allgather, LPT assignment).
    setup_time = MPI.Wtime() - t_start
    t_local_start = MPI.Wtime()
    local_freq, local_tokens = count_in_files(my_files, target_set)
    local_time = MPI.Wtime() - t_local_start

    # Step 7: gather partial results in rank 0.
    partial = (rank, len(my_files), assigned_bytes, local_tokens,
               local_time, setup_time, dict(local_freq))
    gathered = comm.gather(partial, root=0)

    if rank == 0:
        freq_global = Counter()
        stats = []
        for r, n_files, n_bytes, n_tokens, t_local, t_setup, freq in gathered:
            freq_global.update(freq)
            stats.append((r, n_files, n_bytes, n_tokens, t_local, t_setup))

        elapsed = MPI.Wtime() - t_start

        times = [s[4] for s in stats]
        mean_t = sum(times) / len(times)
        li = max(times) / mean_t if mean_t > 0 else float("nan")
        spread = (max(times) - min(times)) / mean_t if mean_t > 0 else float("nan")

        # Sort by count desc, then alphabetically for deterministic ties.
        top_words = sorted(freq_global.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_N]

        results_path = os.path.join(dataset_dir, "mpi2_results.csv")
        with open(results_path, "w", encoding="utf-8") as f:
            f.write("palabra,conteo\n")
            for word in sorted(freq_global):
                f.write(f"{word},{freq_global[word]}\n")

        matches = compare_with_baseline(
            freq_global, os.path.join(dataset_dir, "baseline_results.csv")
        )

        print(f"MPI version 2 (parallel size scan + LPT by file size), p = {size}")
        print(f"Files processed: {sum(s[1] for s in stats)}")
        print(f"Tokens read: {sum(s[3] for s in stats)}")
        print(f"Total occurrences found: {sum(freq_global.values())}\n")

        print("Per-process stats:")
        for r, n_files, n_bytes, n_tokens, t_local, t_setup in stats:
            print(f"  rank {r}: files={n_files}, bytes={n_bytes}, tokens={n_tokens}, "
                  f"setup_time={t_setup:.6f} s, local_time={t_local:.6f} s")

        print(f"\nLOAD_IMBALANCE_LI={li:.4f}")
        print(f"RELATIVE_SPREAD={spread:.4f}")
        print(f"MATCHES_BASELINE={matches}\n")

        print(f"Top {TOP_N} query words in the corpus:")
        for word, count in top_words:
            print(f"  {word}: {count}")

        print(f"\nEXECUTION_TIME={elapsed:.6f}")


if __name__ == "__main__":
    main()