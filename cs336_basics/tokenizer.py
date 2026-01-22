from typing import Iterable, Iterator
from .train_bpe import pretokenize_string, tupled_bytes, split_with_special_tokens

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.merge_ranks = {pair: idx for idx, pair in enumerate(merges)}

    def _bpe(self, token: tuple[bytes]) -> tuple[bytes]:
        if len(token) < 2:
            return token
        while True:
            best_pair: tuple[bytes, bytes] | None = None
            best_rank: int | None = None
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break

            new_tokens: list[bytes] = []
            i = 0
            while i < len(token):
                if i < len(token) - 1 and (token[i], token[i + 1]) == best_pair:
                    new_tokens.append(token[i] + token[i + 1])
                    i += 2
                else:
                    new_tokens.append(token[i])
                    i += 1
            token = tuple(new_tokens)
            if len(token) < 2:
                break
        return token

    def _encode_text_iter(self, text: str) -> Iterator[int]:
        if not text:
            return
        segments = split_with_special_tokens(text, self.special_tokens)
        for segment in segments:
            if segment in self.special_tokens:
                token_id = self.token_to_id[segment.encode("utf-8")]
                yield token_id
                continue
            tokens = pretokenize_string(
                segment,
                r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            )
            for token in tokens:
                token_bytes = token.group(0).encode("utf-8")
                token_tuple = tupled_bytes(token_bytes)
                if len(token_tuple) == 1 and token_tuple[0] in self.token_to_id:
                    yield self.token_to_id[token_tuple[0]]
                    continue
                for subtoken in self._bpe(token_tuple):
                    yield self.token_to_id[subtoken]

    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        # Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges 
        # (in the same format that your BPE training code output) and (optionally) a list of special tokens. 
        pass

    def encode(self, text: str) -> list[int]:
        return list(self._encode_text_iter(text))
            

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # 完全不会Iterator啊
        if not iterable:
            return

        continuation_prefixes: set[str] = set()
        for token in self.special_tokens:
            for i in range(1, len(token)):
                continuation_prefixes.add(token[:i])

        max_prefix_len = max((len(p) for p in continuation_prefixes), default=0)
        buffer = ""

        def emit_match_text(match_text: str) -> Iterator[int]:
            token_bytes = match_text.encode("utf-8")
            token_tuple = tupled_bytes(token_bytes)
            if len(token_tuple) == 1 and token_tuple[0] in self.token_to_id:
                yield self.token_to_id[token_tuple[0]]
                return
            for subtoken in self._bpe(token_tuple):
                yield self.token_to_id[subtoken]

        for chunk in iterable:
            if not chunk:
                continue
            buffer += chunk

            tail_len = 0
            if max_prefix_len > 0:
                scan_len = min(len(buffer), max_prefix_len)
                for i in range(scan_len, 0, -1):
                    if buffer[-i:] in continuation_prefixes:
                        tail_len = i
                        break
            if tail_len:
                safe_text = buffer[:-tail_len]
                tail = buffer[-tail_len:]
            else:
                safe_text = buffer
                tail = ""

            if not safe_text:
                buffer = tail
                continue

            segments = split_with_special_tokens(safe_text, self.special_tokens)
            pending_text = ""
            for idx, segment in enumerate(segments):
                is_last_segment = idx == len(segments) - 1
                if segment in self.special_tokens:
                    if pending_text:
                        for _id in self._encode_text_iter(pending_text):
                            yield _id
                        pending_text = ""
                    yield self.token_to_id[segment.encode("utf-8")]
                    continue

                if not is_last_segment:
                    for _id in self._encode_text_iter(segment):
                        yield _id
                    continue

                # Last segment: avoid splitting a trailing regex token
                prev_match = None
                for match in pretokenize_string(
                    segment,
                ):
                    if prev_match is not None:
                        for _id in emit_match_text(prev_match.group(0)):
                            yield _id
                    prev_match = match
                if prev_match is not None and prev_match.end() == len(segment):
                    pending_text = prev_match.group(0)
                elif prev_match is not None:
                    # 这个分支不会走到
                    for _id in emit_match_text(prev_match.group(0)):
                        yield _id
                    pending_text = ""

            buffer = pending_text + tail

        if buffer:
            for _id in self._encode_text_iter(buffer):
                yield _id

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        return b"".join([self.vocab[id] for id in ids]).decode("utf-8", errors="replace")


if __name__ == "__main__":
    from tests.common import gpt2_bytes_to_unicode
    import os, json
    FIXTURES_PATH = "/home/wangfeiyu/llm-from-scratch-assignment1-basics/tests/fixtures"
    VOCAB_PATH = os.path.join(FIXTURES_PATH, "gpt2_vocab.json")
    MERGES_PATH = os.path.join(FIXTURES_PATH, "gpt2_merges.txt")


    def get_tokenizer_from_vocab_merges_path(
        vocab_path: str | os.PathLike,
        merges_path: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_path) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_path) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        return Tokenizer(vocab, merges, special_tokens)
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"])
    sample_text = "Hello, world! I'm testing the tokenizer. <|endoftext|>"
    encoded = tokenizer.encode(sample_text)
    print("Encoded:", encoded)
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)
