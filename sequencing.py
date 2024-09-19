import numpy as np

def pad_sequence(sequence, target_length, pad_token):
    """Pad a sequence to the target length using the given pad token."""
    return sequence + [pad_token] * (target_length - len(sequence))


def handle_incomplete_target(
    target, max_target_length, pad_token, unk_token, pad_incomplete
):
    """Handle incomplete target sequences by padding or marking as unknown."""
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
    step_by_word=True,
    skip_processed_tokens=False,
    check_length=True,
    pad_incomplete=False,
    pad_token=None,
    unk_token=None,
):
    """
    Create context-target token sequences from tokenized data.

    Args:
        tokenized_data (list): List of tokenized input (as lists of token IDs).
        max_context_length (int): Maximum length for the context.
        max_target_length (int): Maximum length for the target.
        step_by_word (bool): Whether to generate new sequences stepping one token at a time.
        skip_processed_tokens (bool): Whether to skip tokens once they are used.
        check_length (bool): Whether to check if enough tokens remain for context.
        pad_incomplete (bool): Whether to pad incomplete sequences.
        pad_token (int): The token ID used for padding.
        unk_token (int): The token ID used for unknowns (used for incomplete targets).

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

        # Check if there are enough tokens left for the context
        if len(context) < max_context_length and check_length:
            if pad_incomplete and pad_token is not None:
                context = pad_sequence(context, max_context_length, pad_token)
            else:
                break

        target = handle_incomplete_target(
            target, max_target_length, pad_token, unk_token, pad_incomplete
        )

        sequences.append((context, target))

        if step_by_word:
            i += 1
        elif skip_processed_tokens:
            i += max_context_length + max_target_length
        else:
            i += max_context_length

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