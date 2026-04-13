import regex as re
from collections import defaultdict
import time
import multiprocessing as mp
import os
from typing import BinaryIO
import heapq

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class RevPair:
    __slots__ = ("pair",)
    def __init__(self, pair):
        self.pair = pair
    def __lt__(self, other):
        return self.pair > other.pair
    def __eq__(self, other):
        return self.pair == other.pair
    def __le__(self, other):
        return self.pair >= other.pair

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

# helper funciton to get the pair counts from the pretoken count
def get_pair_counts(pretoken_count: defaultdict[bytes, int]) -> defaultdict[tuple[bytes, bytes], int]:
    pair_counts = defaultdict(int)
    for token, count in pretoken_count.items():
        for i in range(len(token) - 1):
            pair_counts[(token[i], token[i+1])] += count
    return pair_counts

# helper funciton to merge a pair of tokens
def merge_pair(pretoken_count: defaultdict[bytes, int], pair: tuple[bytes, bytes]) -> defaultdict[bytes, int]:
    new_pretoken_count = defaultdict(int)
    l, r = pair
    new_token = l + r

    for token_seq, count in pretoken_count.items():
        new_seq = []
        i = 0
        
        while i < len(token_seq):
            if i < len(token_seq) - 1 and token_seq[i] == l and token_seq[i+1] == r:
                new_seq.append(new_token)
                i += 2
            else:
                new_seq.append(token_seq[i])
                i += 1
        new_pretoken_count[tuple(new_seq)] += count
    return new_pretoken_count

def merge_pair_optimized(
    pretoken_count: dict[tuple[bytes, ...], int],
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_index: dict[tuple[bytes, bytes], set],
    pair: tuple[bytes, bytes],
    heap: list,
) -> dict[tuple[bytes, ...], int]:
    l, r = pair
    new_token = l + r
    
    affected = pair_index.pop(pair, set())
    
    for token_seq in affected:
        count = pretoken_count.get(token_seq, 0)
        if count == 0:
            continue
        
        new_seq = []
        i = 0
        while i < len(token_seq):
            if i < len(token_seq) - 1 and token_seq[i] == l and token_seq[i+1] == r:
                if new_seq:
                    old_left = (new_seq[-1], l)
                    pair_counts[old_left] -= count
                    heapq.heappush(heap, (-pair_counts[old_left], RevPair(old_left)))
                    pair_index[old_left].discard(token_seq)

                if i + 2 < len(token_seq):
                    old_right = (r, token_seq[i+2])
                    pair_counts[old_right] -= count
                    heapq.heappush(heap, (-pair_counts[old_right], RevPair(old_right)))
                    pair_index[old_right].discard(token_seq)

                new_seq.append(new_token)

                if len(new_seq) >= 2:
                    new_left = (new_seq[-2], new_token)
                    pair_counts[new_left] += count
                    heapq.heappush(heap, (-pair_counts[new_left], RevPair(new_left)))

                if i + 2 < len(token_seq):
                    new_right = (new_token, token_seq[i+2])
                    pair_counts[new_right] += count
                    heapq.heappush(heap, (-pair_counts[new_right], RevPair(new_right)))

                i += 2
            else:
                new_seq.append(token_seq[i])
                i += 1
        
        new_token_seq = tuple(new_seq)
        del pretoken_count[token_seq]
        pretoken_count[new_token_seq] += count
        
        for i in range(len(new_token_seq) - 1):
            pair_index[(new_token_seq[i], new_token_seq[i+1])].add(new_token_seq)
    
    pair_counts[pair] = 0
    return pretoken_count

# main function to train the BPE
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # initialize the vocab
    vocab = {i: bytes([i]) for i in range(256)}
    cur = 256
    for special_token in special_tokens:
        vocab[cur] = special_token.encode("utf-8")
        cur += 1

    merges = [] # list[tuple[bytes, bytes]]

    # pretokenization --> splitting on special tokens into different chunks
    if special_tokens:
        special_pattern = "|".join(re.escape(st) for st in special_tokens)
        chunks = re.split(special_pattern, text)

    else:
        chunks = [text]
    
    # count the pretokens in each chunk
    pretoken_count = defaultdict(int)
    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            word = match.group().encode("utf-8")
            bytes_word = tuple(bytes([char]) for char in word)
            pretoken_count[bytes_word] += 1
    
    # aactual BPE training loop
    while len(vocab) < vocab_size:
        pair_counts = get_pair_counts(pretoken_count)
        if not pair_counts:
            break
    
        selected_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        selected_token = selected_pair[0] + selected_pair[1]
        
        vocab[cur] = selected_token
        cur += 1
        merges.append(selected_pair)

        pretoken_count = merge_pair(pretoken_count, selected_pair)
    
    return vocab, merges
    
def pretokenize(args: tuple[str, int, int, list[str]]) -> defaultdict[tuple[bytes, int], int]:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    
    if special_tokens:
        special_pattern = "|".join(re.escape(st) for st in special_tokens)
        parts = re.split(special_pattern, chunk)

    else:
        parts = [chunk]
    
    pretoken_count = defaultdict(int)
    for part in parts:
        for match in re.finditer(PAT, part):
            word = match.group().encode("utf-8")
            bytes_word = tuple(bytes([char]) for char in word)
            pretoken_count[bytes_word] += 1
    return pretoken_count

def build_pair_index(pretoken_count):
    pair_index = defaultdict(set)
    for token_seq in pretoken_count:
        for i in range(len(token_seq) - 1):
            pair_index[(token_seq[i], token_seq[i+1])].add(token_seq)
    return pair_index

def train_bpe_optimized(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    t0 = time.time()

    # initialize the vocab
    vocab = {i: bytes([i]) for i in range(256)}
    cur = 256
    for special_token in special_tokens:
        vocab[cur] = special_token.encode("utf-8")
        cur += 1

    merges = [] # list[tuple[bytes, bytes]]

    # pretokenization --> splitting on special tokens into different chunks
    t1 = time.time()
    num_processes = mp.cpu_count()

    pretoken_count = defaultdict(int)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

    chunk_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    
    with mp.Pool(num_processes) as pool:
        results = pool.map(pretokenize, chunk_args)
    
    for chunk in results:
        for token, count in chunk.items():
            pretoken_count[token] += count

    print(f"Time taken for pretokenization: {time.time() - t1} seconds")

    # actual BPE training loop
    t2 = time.time()
    pair_counts = get_pair_counts(pretoken_count)
    pair_index = build_pair_index(pretoken_count)

    heap = [(-count, RevPair(pair)) for pair, count in pair_counts.items()]
    heapq.heapify(heap)

    while len(vocab) < vocab_size:
        # lazy deletion of stale entries
        while heap:
            neg_count, rev = heap[0]
            if pair_counts.get(rev.pair, 0) == -neg_count:
                break
            heapq.heappop(heap)

        if not heap:
            break

        neg_count, rev = heapq.heappop(heap)
        selected_pair = rev.pair
        selected_token = selected_pair[0] + selected_pair[1]
        vocab[cur] = selected_token
        cur += 1
        merges.append(selected_pair)

        pretoken_count = merge_pair_optimized(
            pretoken_count, pair_counts, pair_index, selected_pair, heap
        )

    print(f"Time taken for BPE training: {time.time() - t2:.2f}s")
    return vocab, merges