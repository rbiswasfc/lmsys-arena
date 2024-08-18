from dataclasses import dataclass

import torch
from transformers import DataCollatorWithPadding


@dataclass
class LMSYSCollator(DataCollatorWithPadding):
    """
    Data collector for LMSYS - Chatbot Arena Human Preference Predictions task
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"

    def __call__(self, features):
        labels = None
        if "label" in features[0].keys():
            labels = [feature["label"] for feature in features]

        features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]

        batch = self.tokenizer.pad(
            features,
            padding="longest",
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        if labels is not None:
            batch["labels"] = labels

        tensor_keys = [
            "input_ids",
            "attention_mask",
        ]

        for key in tensor_keys:
            batch[key] = torch.tensor(batch[key], dtype=torch.int64)

        if labels is not None:
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.float32)

        return batch


def show_batch(batch, tokenizer, n_examples=8, task="training", print_fn=print):
    bs = batch["input_ids"].size(0)
    print_fn(f"batch size: {bs}")
    print_fn(f"shape of input_ids: {batch['input_ids'].shape}")

    n_examples = min(n_examples, bs)
    print_fn(f"Showing {n_examples} from a {task} batch...")

    print_fn("\n\n")
    for idx in range(n_examples):
        print_fn(f"=== Example {idx+1} ===")
        print_fn(f"Input:\n\n{tokenizer.decode(batch['input_ids'][idx], skip_special_tokens=False)}")

        if "infer" not in task.lower():
            print_fn("--" * 20)
            labels = batch["labels"][idx]
            print_fn(f"Label: {labels}")
        print_fn("~~" * 40)
