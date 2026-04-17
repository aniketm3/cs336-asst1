from cs336_basics.train_bpe import train_bpe

import pickle
from typing import Iterable
from collections import defaultdict, deque
import re
from __future__ import annotations


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        self.pattern = re.compile(PAT)
        self.special_token_pattern = re.compile(
            "(" + "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)) + ")"
            if self.special_tokens else "(?!)"
        )

        self.merge_rank = {merge: i for i, merge in enumerate(self.merges)} # (token1, token2) -> rank
        self.token_id_map = {v: k for k, v in self.vocab.items()} # token -> id

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None) -> Tokenizer:
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_path, "rb") as f:
            merges = pickle.load(f)
        
        return cls(vocab, merges, special_tokens)

    # encoding for the given text
    def encode_token(self, text: str) -> list[int]:
        tokens = [bytes([b]) for b in text.encode("utf-8")]

        # merge tokens until no more merges are possible
        while True:
            best = None
            best_rank = float("inf")
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                rank = self.merge_rank.get(pair, None)

                if rank is not None and rank < best_rank:
                    best = pair
                    best_rank = rank
                
            if best is None:
                break
        
            new_tokens = []
            i = 0
            # merge tokens if they are consecutive
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best[0] and tokens[i+1] == best[1]:
                    new_tokens.append(best[0] + best[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            tokens = new_tokens
        
        return [self.token_id_map[token] for token in tokens]
    
    # dsipatcher for encoding on the given text --> handles special tokens and splitting per pattern
    def encode(self, text: str) -> list[int]:
        ret_ids = []

        # split text into segments based on special tokens
        for segment in self.special_token_pattern.split(text):
            # handle special tokens (they are not merged)
            if segment in self.special_tokens:
                ret_ids.append(self.token_id_map[segment.encode("utf-8")])
        
            # handle normal tokens (they are merged)
            else:
                for sub_segment in self.pattern.finditer(segment):
                    encoding = self.encode_token(sub_segment.group(0))  
                    if encoding:
                        ret_ids.extend(encoding)
    
        return ret_ids

    def encode_iterable(self, iterable: Iterable[str]) -> list[int]:
        for text in iterable:
            encoding = self.encode(text)
            yield from encoding
        
    def decode(self, ids: list[int]) -> str:
        decoding = [self.vocab[id] for id in ids] # currently all bytes
        return b"".join(decoding).decode("utf-8")
    

