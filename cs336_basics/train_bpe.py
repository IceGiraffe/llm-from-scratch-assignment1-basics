from dataclasses import dataclass
import os
from typing import BinaryIO
import regex as re
from concurrent.futures import ProcessPoolExecutor


# copied from pretokenization_example.py
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[str],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    All chunks start with spilit_special_token except the first chunk.
    """
    assert all(
        isinstance(tokens, bytes) for tokens in split_special_tokens
    ), "Must represent special tokens as bytestrings"

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

            # To find one of multiple possible special tokens, we just look for the first one
            found_at = -1
            for token in split_special_tokens:
                pos = mini_chunk.find(token)
                if pos != -1:
                    found_at = pos
                    break
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def split_with_special_tokens(chunk: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [chunk]
    # Prefer longer tokens when they overlap.
    escaped_tokens = [
        re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)
    ]
    pattern = "(" + "|".join(escaped_tokens) + ")"
    parts = re.split(pattern, chunk)
    # print(parts)
    print(special_tokens[0] in parts)
    return [part for part in parts if part]


def pretokenize_string(
    segment: str,
    regex_pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
) -> list[bytes]:
    # Simple pre-tokenization by splitting on whitespace
    tokens = re.finditer(regex_pattern, segment)
    return tokens


def pretokenize_chunk(
    segments: list[str],
    special_tokens: list[str],
    regex_pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
) -> dict[tuple[bytes], int]:
    pretokenized: dict[tuple[bytes], int] = {}
    for segment in segments:
        if segment in special_tokens:
            token_bytes = segment.encode("utf-8")
            token_tuple = (token_bytes,)
            if token_tuple in pretokenized:
                pretokenized[token_tuple] += 1
            else:
                pretokenized[token_tuple] = 1
            continue
        tokens = pretokenize_string(segment, regex_pattern)
        bytes_list = [token.group(0).encode("utf-8") for token in tokens]
        tupled_list = [tupled_bytes(b) for b in bytes_list]
        for tuple_bytes in tupled_list:
            if tuple_bytes in pretokenized:
                pretokenized[tuple_bytes] += 1
            else:
                pretokenized[tuple_bytes] = 1
    return pretokenized


def convert_pretokenized(tokens: list[:]) -> dict[tuple[bytes], int]:
    pass


def tupled_bytes(b: bytes) -> tuple[bytes]:
    return tuple(bytes([x]) for x in b)


def initialize_vocab(
    special_tokens: list[str],
) -> dict[int, bytes]:
    # initialize our vocabulary with our special tokens and the 256 byte
    vocab: dict[int, bytes] = {}
    for idx, token in enumerate(special_tokens):
        vocab[idx] = token.encode("utf-8")
    for i in range(256):
        vocab[len(special_tokens) + i] = bytes([i])
    return vocab


def update_vocab_with_merge(
    vocab: dict[int, bytes],
    token1: bytes,
    token2: bytes,
) -> int:
    new_token = token1 + token2
    new_id = max(vocab.keys()) + 1
    vocab[new_id] = new_token
    return new_id


@dataclass
class PairCount:
    pair: tuple[bytes, bytes]
    count: int

    def __lt__(self, other: "PairCount") -> bool:
        if self.count == other.count:
            return self.pair < other.pair  # lexicographically greater
        return self.count < other.count


def bpe_merge(
    pretokenized_chunk: dict[tuple[bytes], int],
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    target_vocab_size: int,
) -> None:
    from collections import Counter

    while len(vocab) < target_vocab_size:
        pair_counts: list[PairCount] = []

        # Count frequency of each adjacent token pair
        pair_frequency: Counter[tuple[bytes, bytes]] = Counter()
        for token_tuple, freq in pretokenized_chunk.items():
            for i in range(len(token_tuple) - 1):
                pair = (token_tuple[i], token_tuple[i + 1])
                pair_frequency[pair] += freq
        for pair, count in pair_frequency.items():
            pair_counts.append(PairCount(pair, count))
        pair_counts.sort(reverse=True)

        most_frequent_pair, _ = pair_counts[0].pair, pair_counts[0].count
        # print(most_frequent_pair, _)
        token1, token2 = most_frequent_pair

        # Create new token and update vocab
        new_token_id = update_vocab_with_merge(vocab, token1, token2)
        # print(new_token_id)
        merges.append((token1, token2))

        # Update pretokenized_chunk with the new merged token
        new_pretokenized_chunk: dict[tuple[bytes], int] = {}
        for token_tuple, freq in pretokenized_chunk.items():
            new_token_list = []
            i = 0
            while i < len(token_tuple):
                if (
                    i < len(token_tuple) - 1
                    and (token_tuple[i], token_tuple[i + 1]) == most_frequent_pair
                ):
                    new_token_list.append(token1 + token2)
                    i += 2
                else:
                    new_token_list.append(token_tuple[i])
                    i += 1
            new_token_tuple = tuple(new_token_list)
            if new_token_tuple in new_pretokenized_chunk:
                new_pretokenized_chunk[new_token_tuple] += freq
            else:
                new_pretokenized_chunk[new_token_tuple] = freq

        pretokenized_chunk = new_pretokenized_chunk


def worker(args) -> dict[tuple[bytes], int]:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        # split chunk into normal segments and preserved special tokens
        segments = split_with_special_tokens(chunk, special_tokens)

        # pre-tokenization
        pretokenized_chunk: dict[tuple[bytes], int] = pretokenize_chunk(
            segments, special_tokens
        )
        return pretokenized_chunk


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    print("Training BPE with the following parameters:")
    print(f"  input_path: {input_path}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  special_tokens: {special_tokens}")

    # firt find chunk boundaries
    f = open(input_path, "rb")
    num_processes = 16
    boundaries = find_chunk_boundaries(
        f, num_processes, [token.encode("utf-8") for token in special_tokens]
    )
    f.close()
    print("Chunk boundaries:", boundaries)
    num_chunks = max(1, len(boundaries) - 1)
    max_workers = min(num_processes, num_chunks)

    # 默认使用本机的CPU核数，需要指定吗 max_workers
    # 避免默认 CPU 核数导致的启动开销
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = list(
            ex.map(
                worker,
                [
                    (input_path, start, end, special_tokens)
                    for start, end in zip(boundaries[:-1], boundaries[1:])
                ],
            )
        )

    # merge results, 一开始是这里写错了
    pretokenized_chunk: dict[tuple[bytes], int] = {}
    for partial in results:
        for token_tuple, count in partial.items():
            if token_tuple in pretokenized_chunk:
                pretokenized_chunk[token_tuple] += count
            else:
                pretokenized_chunk[token_tuple] = count

    print("Total unique pre-tokenized items:", len(pretokenized_chunk))

    vocab: dict[int, bytes] = initialize_vocab(special_tokens)
    print("Initial vocab size:", len(vocab))

    merges: list[tuple[bytes, bytes]] = []
    bpe_merge(
        pretokenized_chunk,
        vocab,
        merges,
        vocab_size,
    )
    print("Final vocab size:", len(vocab))
    print("Number of merges:", len(merges))

    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe(
        input_path="data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )
    print("Trained vocabulary:")
    for token_id, token_bytes in vocab.items():
        print(f"{token_id}: {token_bytes}")
    print("\nTrained merges:")
    for token1, token2 in merges:
        print(f"({token1}, {token2})")
