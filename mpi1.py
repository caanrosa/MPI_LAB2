# MPI version 1: word counting with broadcast and static file distribution.

# Flow:
#  1. rank 0 reads consulta.txt
#  2. rank 0 broadcasts the query words to all processes (comm.bcast)
#  3. rank 0 lists the file_*.txt files of the corpus
#  4. files are split statically into contiguous blocks (comm.scatter)
#  5. each process counts the query words in its own files
#  6. partial results are gathered in rank 0 (comm.gather)
#  7. rank 0 merges them, prints per-process stats and the top 10


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


def list_corpus_files(dataset_dir):
    """Return the sorted list of file_*.txt paths."""
    names = sorted(
        n for n in os.listdir(dataset_dir)
        if n.startswith("file_") and n.endswith(".txt")
        and os.path.isfile(os.path.join(dataset_dir, n))
    )
    return [os.path.join(dataset_dir, n) for n in names]


def split_static(items, parts):
    
    # Static distribution: contiguous blocks whose sizes differ by at most 1. Only the NUMBER of files is balanced, not the amount of work.
    
    
    base, extra = divmod(len(items), parts)
    blocks, start = [], 0
    for i in range(parts):
        end = start + base + (1 if i < extra else 0)
        blocks.append(items[start:end])
        start = end
    return blocks


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

    # Steps 1-3: rank 0 reads the query and lists the files.
    query_words = None
    file_blocks = None
    if rank == 0:
        query_words = load_query(os.path.join(dataset_dir, CONSULTA_NAME))
        all_files = list_corpus_files(dataset_dir)
        file_blocks = split_static(all_files, size)

    # Step 2: broadcast the query words.
    query_words = comm.bcast(query_words, root=0)
    target_set = set(query_words)

    # Step 4: static distribution of files.
    my_files = comm.scatter(file_blocks, root=0)

    # Step 5: local counting (timed per process).
    # setup_time = everything before counting starts (read query, list files,
    # bcast, scatter, waiting for rank 0).
    setup_time = MPI.Wtime() - t_start
    t_local_start = MPI.Wtime()
    local_freq, local_tokens = count_in_files(my_files, target_set)
    local_time = MPI.Wtime() - t_local_start
 
    # Step 6: gather partial results in rank 0.
    partial = (rank, len(my_files), local_tokens, local_time,
               setup_time, dict(local_freq))
    gathered = comm.gather(partial, root=0)
 
    # Step 7: rank 0 builds the global result.
    if rank == 0:
        freq_global = Counter()
        stats = []
        for r, n_files, n_tokens, t_local, t_setup, freq in gathered:
            freq_global.update(freq)
            stats.append((r, n_files, n_tokens, t_local, t_setup))
 
        elapsed = MPI.Wtime() - t_start
 
        times = [s[3] for s in stats]
        mean_t = sum(times) / len(times)
        li = max(times) / mean_t if mean_t > 0 else float("nan")
        spread = (max(times) - min(times)) / mean_t if mean_t > 0 else float("nan")
 
        # Sort by count desc, then alphabetically for deterministic ties.
        top_words = sorted(freq_global.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_N]
 
        results_path = os.path.join(dataset_dir, "mpi1_results.csv")
        with open(results_path, "w", encoding="utf-8") as f:
            f.write("palabra,conteo\n")
            for word in sorted(freq_global):
                f.write(f"{word},{freq_global[word]}\n")
 
        matches = compare_with_baseline(
            freq_global, os.path.join(dataset_dir, "baseline_results.csv")
        )
 
        print(f"MPI version 1 (static distribution), p = {size}")
        print(f"Files processed: {sum(s[1] for s in stats)}")
        print(f"Tokens read: {sum(s[2] for s in stats)}")
        print(f"Total occurrences found: {sum(freq_global.values())}\n")
 
        print("Per-process stats:")
        for r, n_files, n_tokens, t_local, t_setup in stats:
            print(f"  rank {r}: files={n_files}, tokens={n_tokens}, "
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