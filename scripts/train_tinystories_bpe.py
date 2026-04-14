import cProfile
import pickle
from cs336_basics.tokenizer import train_bpe_optimized
import cProfile
import pstats
import tracemalloc


if __name__ == "__main__":
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    input_path = "data/TinyStoriesV2-GPT4-train.txt"

    # tracemalloc.start()

    with cProfile.Profile() as pr:
        vocab, merges = train_bpe_optimized(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens
        )

    # memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Memory usage: {current / 1024 / 1024} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024} MB")


    # profiling
    p = pstats.Stats(pr)
    p.sort_stats("cumulative")
    p.print_stats(20)


    # longest token
    longest_token = max(vocab.values(), key=len)
    print(f"Longest token: {longest_token}")


    # random stats
    print(f"Vocab size: {len(vocab)}")
    print(f"Merges size: {len(merges)}")


    # saving the vocab and merges
    with open("data/tinystories_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open("data/tinystories_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

