from tokenizer import BPETokenizer
from sequencing import create_sequences, numperize

dataset = [
    "some text",
    "wow, this is some more text",
    "this is a third text",
    "maybe some other text",
    "or words",
    "and more words",
]

DEFAULT_SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>"]

tokenizer = BPETokenizer(special_tokens=DEFAULT_SPECIAL_TOKENS)
tokenizer.fit(dataset)
tokenized_data = [tokenizer.encode(text).ids for text in dataset]

flat_tokenized_data = [token_id for sublist in tokenized_data for token_id in sublist]

if not flat_tokenized_data:
    raise ValueError(
        "The flattened tokenized data is empty. Please check your tokenizer and dataset."
    )

sequences = create_sequences(
    tokenized_data=flat_tokenized_data,
    max_context_length=10,
    max_target_length=1,
    step_by_word=True,
    pad_token=1,
    unk_token=3,
)

contexts_array, targets_array = numperize(sequences)

for i, (context, target) in enumerate(sequences):
    print(f"Sequence {i}:")
    print(f"Context: {contexts_array[i]}")
    print(f"Target: {targets_array[i]}")
    print(
        f"Decoded context: {[tokenizer.decode([token_id]) for token_id in contexts_array[i]]}"
    )
    print(
        f"Decoded target: {[tokenizer.decode([token_id]) for token_id in targets_array[i]]}"
    )
    print()
