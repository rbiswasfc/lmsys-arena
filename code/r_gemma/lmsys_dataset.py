from copy import deepcopy

from datasets import Dataset
from transformers import AutoTokenizer


def get_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.backbone_path,
        use_fast=cfg.model.tokenizer.use_fast,
        padding_side=cfg.model.tokenizer.padding_side,
        truncation_side=cfg.model.tokenizer.truncation_side,
    )

    if tokenizer.eos_token == "":
        tokenizer.add_special_tokens({"eos_token": "</s>"})
        tokenizer.eos_token = "</s>"

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def truncate_text(text, max_char=4096):
    if len(text) <= max_char:
        return text

    max_char = max_char // 2
    text = text[:max_char] + "\n\n<unused1>\n\n" + text[-max_char:]
    return text


class LMSYSDataset:
    """
    Dataset class for LMSYS - Chatbot Arena Human Preference Predictions competition
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = get_tokenizer(cfg)

    def tokenize_function(self, examples):
        tx = self.tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=self.cfg.model.max_length,
            return_length=True,
            add_special_tokens=True,
        )
        return tx

    def wrap_content(self, content, role):
        wrapped_content = f"<start_of_turn>{role}\n{content.strip()}<end_of_turn>\n"
        return wrapped_content

    def apply_chat_template(self, messages):
        text = ""

        prompt = ""
        response_a = ""
        response_b = ""

        for message in messages:
            role = message["role"]

            if role == "User":
                prompt += f"## {message['content']}\n"
            elif role == "Assistant A":
                response_a += f"## {message['content']}\n"
            elif role == "Assistant B":
                response_b += f"## {message['content']}\n"
            else:
                raise ValueError(f"Invalid role: {role}")

        prompt = self.wrap_content(truncate_text(prompt), role="user")
        response_a = self.wrap_content(truncate_text(response_a), role="assistant a")
        response_b = self.wrap_content(truncate_text(response_b), role="assistant b")

        text = f"{prompt}\n{response_a}\n{response_b}\n"
        return text

    def preprocess_function(self, df):
        formatted_texts = []

        for _, row in df.iterrows():
            prompts = row["prompt"]
            response_a = row["response_a"]
            response_b = row["response_b"]

            # keep first 2 and last 2 prompts ---
            num_turns_each_end = 2
            if len(prompts) > num_turns_each_end * 2:
                prompts = prompts[:num_turns_each_end] + prompts[-num_turns_each_end:]
                response_a = response_a[:num_turns_each_end] + response_a[-num_turns_each_end:]
                response_b = response_b[:num_turns_each_end] + response_b[-num_turns_each_end:]

            messages = []
            for turn, (p, a, b) in enumerate(zip(prompts, response_a, response_b)):
                messages.append(
                    {
                        "Turn": turn + 1,
                        "role": "User",
                        "content": truncate_text(p, max_char=self.cfg.model.tokenizer.max_char),
                    }
                )
                messages.append(
                    {
                        "Turn": turn + 1,
                        "role": "Assistant A",
                        "content": truncate_text(a, max_char=self.cfg.model.tokenizer.max_char),
                    }
                )
                messages.append(
                    {
                        "Turn": turn + 1,
                        "role": "Assistant B",
                        "content": truncate_text(b, max_char=self.cfg.model.tokenizer.max_char),
                    }
                )

            text = self.apply_chat_template(messages)
            formatted_texts.append(text)

        df["text"] = formatted_texts
        return df

    def get_dataset(self, df):
        """use this function to get the dataset

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            Dataset: HF Dataset object with tokenized inputs and labels
        """

        df = deepcopy(df)
        df = self.preprocess_function(df)
        task_dataset = Dataset.from_pandas(df)
        remove_columns = [col for col in df.columns if col not in ["id", "label"]]

        task_dataset = task_dataset.map(
            self.tokenize_function, batched=True, num_proc=self.cfg.model.num_proc, remove_columns=remove_columns
        )

        return task_dataset