import numpy as np

def select_examples(feedback_dataset, evaluation_details, hp):
    """
    Select examples to guide data augmentation based on model evaluation.

    Args:
        feedback_dataset: Dataset containing the examples corresponding
            to the evaluation predictions.
        evaluation_details: Dictionary containing:
            - labels: Ground-truth labels
            - preds: Predicted labels
            - probs: Predicted probabilities for the positive class
            - threshold: Classification threshold
        hp: Hyperparameter dictionary containing:
            - selection_strategy: Selection strategy
            - num_examples: Maximum number of examples to select

    Returns:
        selected_examples: Selected examples from feedback_dataset.
    """
    selected_examples = []

    # TODO

    labels = np.asarray(evaluation_details["labels"])
    preds = np.asarray(evaluation_details["preds"])
    probs = np.asarray(evaluation_details["probs"])
    threshold = evaluation_details["threshold"]

    strategy = hp.get("selection_strategy", "incorrect")
    num_examples = hp.get("num_examples", len(labels))

    # Identify candidate examples
    incorrect = labels != preds
    false_positive = (labels == 0) & (preds == 1)
    false_negative = (labels == 1) & (preds == 0)

    if strategy == "incorrect":
        scores = incorrect.astype(float)
        candidates = np.where(incorrect)[0]
        priorities = np.ones(len(candidates))

    elif strategy == "false_positive":
        candidates = np.where(false_positive)[0]
        priorities = probs[candidates]

    elif strategy == "false_negative":
        candidates = np.where(false_negative)[0]
        priorities = 1 - probs[candidates]

    elif strategy == "uncertain":
        candidates = np.arange(len(labels))
        priorities = np.abs(probs[candidates] - threshold)

    elif strategy == "low_confidence":
        candidates = np.arange(len(labels))
        priorities = np.abs(probs[candidates] - 0.5)

    elif strategy == "random":
        candidates = np.arange(len(labels))
        priorities = np.random.random(len(candidates))

    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")

    # Sort by priority (lower means more important)
    if strategy == "random":
        np.random.shuffle(candidates)
    else:
        candidates = candidates[np.argsort(priorities)]

    # Select top examples
    selected_indices = candidates[:num_examples]

    # Return corresponding examples
    if hasattr(feedback_dataset, "iloc"):
        selected_examples = feedback_dataset.iloc[selected_indices]
    else:
        selected_examples = [feedback_dataset[i] for i in selected_indices]

    return selected_examples
