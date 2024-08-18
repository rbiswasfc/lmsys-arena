from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.models.llama.modeling_llama import LlamaModel, LlamaPreTrainedModel


class LMSYSHead(nn.Module):
    def __init__(self, in_features, num_labels=3):
        super().__init__()
        self.head = nn.Linear(in_features, num_labels, bias=False)

    def forward(self, x, **kwargs):
        logits = self.head(x[:, -1])
        return logits


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target, weights=None):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class LMSYSAttentionHead(nn.Module):
    def __init__(self, in_features, num_labels, n_head=32, dropout=0.0, bias=False):
        super().__init__()

        n_embd = in_features
        assert n_embd % n_head == 0

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.tok_projection = nn.Linear(in_features, num_labels, bias=False)

        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        assert self.flash, "Flash Attention requires PyTorch >= 2.0"

    def forward(self, x, attention_mask, **kwargs):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, dtype=x.dtype)

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=self.dropout if self.training else 0, is_causal=False
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        feature = y[:, -1] + x[:, -1]
        logits = self.tok_projection(feature)

        return logits


class LLaMaForLMSYS(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)

        # Enable gradient checkpointing ---
        # self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        self.classification_head = LMSYSHead(in_features=config.hidden_size, num_labels=config.num_labels)
        self.loss_fn = DenseCrossEntropy()
        self.post_init()

    def reinit_attn(self):
        # Reinitialize Attention weights (TODO: be aware of peft wrapping)
        scale = 1e-4  # ---
        for name, param in self.classification_head.named_parameters():
            if "c_proj" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
            elif "c_attn" in name:
                nn.init.zeros_(param.data)

        # Scale the parameters by 1e-2 ---
        if "c_proj" in name:
            param.data.mul_(scale)

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
        logits = self.classification_head(hidden_states)  # (bs, num_labels)

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
