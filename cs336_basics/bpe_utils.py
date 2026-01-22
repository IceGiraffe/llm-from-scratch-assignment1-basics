from typing import BinaryIO
import regex as re
import os


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



def tupled_bytes(b: bytes) -> tuple[bytes]:
    return tuple(bytes([x]) for x in b)