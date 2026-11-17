def select_examples(feedback_dataset, evaluation_details, hp):
    """
    identify examples to help guide the augmentation, informed by evaluation details

    e.g., using incorrect, low confidence, false positive examples, etc.

    format for evalulation_details = {
                    "labels": list,
                    "preds": list,
                    "probs": list,
                    "threshold": float
                }
                
    returns: selected_examples
    """
    selected_examples = []

    # TODO

    return selected_examples
