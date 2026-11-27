import numpy as np
from torch.utils.data import Subset

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
    labels = np.asarray(evaluation_details["labels"])
    preds = np.asarray(evaluation_details["preds"])
    probs = np.asarray(evaluation_details["probs"])
    threshold = evaluation_details["threshold"]

    strategy = hp.selection_strategy

    # Identify candidate examples
    incorrect = labels != preds
    false_positive = (labels == 0) & (preds == 1)
    false_negative = (labels == 1) & (preds == 0)

    if strategy == "incorrect":
        candidates = np.where(incorrect)[0]
        priorities = np.ones(len(candidates))

    # only select positives for now
    #elif strategy == "false_positive":
        #candidates = np.where(false_positive)[0]
        #priorities = 1 - probs[candidates]

    elif strategy == "false_negative":
        candidates = np.where(false_negative)[0]
        priorities = probs[candidates]

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

    # Only positive anchors for counterfactual negative generation
    candidates = candidates[labels[candidates] == 1]

    # ensure fixed feedback pool while still allowing the actual training dataset to grow
    if isinstance(feedback_dataset, Subset):
        return [feedback_dataset.dataset.get_raw_example(feedback_dataset.indices[i]) for i in candidates]

    return [feedback_dataset.get_raw_example(i) for i in candidates]
