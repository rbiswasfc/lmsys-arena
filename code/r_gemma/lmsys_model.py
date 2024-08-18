from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.models.gemma2.modeling_gemma2 import Gemma2Model, Gemma2PreTrainedModel


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target, weights=None):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class LMSYSHead(nn.Module):
    def __init__(self, in_features, num_labels=3):
        super().__init__()
        self.head = nn.Linear(in_features, num_labels, bias=False)

    def forward(self, x, **kwargs):
        logits = self.head(x[:, -1])
        return logits


class GemmaForLMSYS(Gemma2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Gemma2Model(config)

        self.classification_head = LMSYSHead(in_features=config.hidden_size, num_labels=config.num_labels)
        # self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.03)
        self.loss_fn = DenseCrossEntropy()

        self.post_init()

    def reinit_head(self):
        for name, param in self.classification_head.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param.data)
            elif "head" in name:
                print(f"re-init {name}")
                nn.init.xavier_uniform_(param.data)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]  # (bs, seq_len, dim)
        # print(hidden_states)
        logits = self.classification_head(hidden_states)  # (bs, num_labels)
        # print(logits)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device).reshape(-1)
            loss = self.loss_fn(logits, labels)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
        )
