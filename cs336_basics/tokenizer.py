import regex as re
from collections import defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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

def merge_pair_optimized(pretoken_count: defaultdict[bytes, int], pair_counts: defaultdict[tuple[bytes, bytes], int], pair: tuple[bytes, bytes]) -> defaultdict[bytes, int]:
    new_pretoken_count = defaultdict(int)
    l, r = pair
    new_token = l + r

    for token_seq, count in pretoken_count.items():
        new_seq = []
        i = 0
        
        while i < len(token_seq):
            if i < len(token_seq) - 1 and token_seq[i] == l and token_seq[i+1] == r:
                if new_seq:
                    old_left_pair = (new_seq[-1], l)
                    pair_counts[old_left_pair] -= count

                if i + 2 < len(token_seq):
                    old_right_pair = (r, token_seq[i+2])
                    pair_counts[old_right_pair] -= count

                new_seq.append(new_token)
                
                # adding the new pair counts for the neighboring pairs
                if len(new_seq) >= 2:
                    new_left_pair = (new_seq[-2], new_token)
                    pair_counts[new_left_pair] += count
                
                if i + 2 < len(token_seq):
                    new_right_pair = (new_token, token_seq[i+2])
                    pair_counts[new_right_pair] += count
                
                i += 2
            
            else:
                new_seq.append(token_seq[i])
                i += 1
        new_pretoken_count[tuple(new_seq)] += count
    
    pair_counts[pair] = 0
    return new_pretoken_count

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
    
def train_bpe_optimized(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

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
    pair_counts = get_pair_counts(pretoken_count)

    while len(vocab) < vocab_size and pair_counts:
        selected_pair = max((p for p in pair_counts if pair_counts[p] > 0), key=lambda p: (pair_counts[p], p))

        selected_token = selected_pair[0] + selected_pair[1]
        vocab[cur] = selected_token
        cur += 1
        merges.append(selected_pair)

        pretoken_count = merge_pair_optimized(pretoken_count, pair_counts, selected_pair)

    return vocab, merges
    