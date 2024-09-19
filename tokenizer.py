from tokenizers import (
    Tokenizer as HFTokenizer,
    models,
    trainers,
    pre_tokenizers,
    decoders,
)

DEFAULT_SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>"]


class BPETokenizer:
    def __init__(
        self, special_tokens=DEFAULT_SPECIAL_TOKENS, vocab_size=30000, min_frequency=2
    ):
        self.tokenizer = HFTokenizer(models.BPE(unk_token=special_tokens[-1]))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        self.tokenizer.decoder = decoders.ByteLevel()
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens

    def fit(self, texts):
        """
        Train the BPE tokenizer using the provided texts.
        :param texts: Iterable of text strings to train on
        """
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
        )
        self.tokenizer.train_from_iterator(texts, trainer)

    def encode(self, text):
        """
        Encode a given text into tokens and token IDs.
        :param text: Text to encode
        :return: Encoded output with token IDs
        """
        return self.tokenizer.encode(text)

    def decode(self, ids):
        """
        Decode a list of token IDs back into text.
        :param ids: List of token IDs
        :return: Decoded text string
        """
        return self.tokenizer.decode(ids)

    def save(self, path):
        """
        Save the trained tokenizer to the specified path.
        :param path: File path to save the tokenizer
        """
        self.tokenizer.save(path)

    def load(self, path):
        """
        Load a tokenizer from the specified file path.
        :param path: File path to load the tokenizer from
        """
        self.tokenizer = HFTokenizer.from_file(path)

    def get_vocab_size(self):
        """
        Get the current vocabulary size of the tokenizer.
        :return: Vocabulary size
        """
        return len(self.tokenizer.get_vocab())

    def get_token_id(self, token):
        """
        Get the ID of a specific token.
        :param token: Token string
        :return: Token ID
        """
        return self.tokenizer.token_to_id(token)

    def __len__(self):
        return len(self.tokenizer.get_vocab())

    def __getitem__(self, item):
        return self.tokenizer.token_to_id(item)

    def __contains__(self, item):
        return item in self.tokenizer.get_vocab()
