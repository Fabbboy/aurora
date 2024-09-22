import numpy as np


def pad_sequence(sequence, target_length, pad_token):
    """Pad a sequence to the target length using the given pad token."""
    return sequence + [pad_token] * (target_length - len(sequence))


def handle_incomplete_target(
    target, max_target_length, pad_token, unk_token, pad_incomplete=True
):
    """
    Handle incomplete target sequences by padding or marking as unknown.

    Args:
        target (list): The target sequence to handle.
        max_target_length (int): Maximum length for the target.
        pad_token (int): The token ID used for padding.
        unk_token (int): The token ID used for unknowns.
        pad_incomplete (bool): Whether to pad incomplete targets.

    Returns:
        list: The processed target sequence.
    """
    if len(target) < max_target_length:
        if pad_incomplete:
            target = pad_sequence(target, max_target_length, pad_token)
        else:
            target = (
                [unk_token] * max_target_length if unk_token is not None else target
            )
    return target


def create_sequences(
    tokenized_data,
    max_context_length,
    max_target_length,
    skip_processed=False,
    pad_token=1,
    unk_token=3,
):
    """
    Create context-target token sequences from tokenized data with fixed behaviors:
    - Always check if there are enough tokens for the context.
    - Always pad incomplete sequences.
    - Do not skip already processed tokens.

    Args:
        tokenized_data (list): Flat list of tokenized input (as token IDs).
        max_context_length (int): Maximum length for the context.
        max_target_length (int): Maximum length for the target.
        pad_token (int): The token ID used for padding.
        unk_token (int): The token ID used for unknowns.

    Returns:
        list: A list of context-target pairs.
    """
    sequences = []
    i = 0

    while i < len(tokenized_data):
        # Define context and target
        context = tokenized_data[i : i + max_context_length]
        target = tokenized_data[
            i + max_context_length : i + max_context_length + max_target_length
        ]

        # Always check if there are enough tokens left for the context
        if len(context) < max_context_length:
            # Always pad incomplete context sequences
            context = pad_sequence(context, max_context_length, pad_token)

        # Always handle incomplete targets by padding
        target = handle_incomplete_target(
            target, max_target_length, pad_token, unk_token, pad_incomplete=True
        )

        sequences.append((context, target))

        i += max_context_length if skip_processed else 1

    return sequences


def numperize(sequences, dtype=np.int32):
    """
    Convert a list of (context, target) tuples into separate NumPy arrays.

    Args:
        sequences (list): List of tuples where each tuple is (context, target).
        dtype (data-type, optional): Desired data-type for the arrays. Default is np.int32.

    Returns:
        tuple: A tuple containing two NumPy arrays (contexts_array, targets_array).
    """
    # Unzip the sequences into two separate lists
    contexts, targets = zip(*sequences)  # This creates two tuples

    # Convert the tuples to NumPy arrays
    contexts_array = np.array(contexts, dtype=dtype)
    targets_array = np.array(targets, dtype=dtype)

    return contexts_array, targets_array
