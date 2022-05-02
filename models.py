from transformers import GPT2PreTrainedModel, GPT2Model
from transformers.modeling_utils import SequenceSummary
#from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModelOutput
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple

# here I'll include everything i need to overwrite the stuff from the distil GPT-2;
# I'll need the quantile reward function
# I'll need the separate output heads;
# I'll need the forward sampling code; with CVaR

# use the define the model class, like they do with the custom loss function.

# taken from GPT2DoubleHeadsModel

class GPT2CustomDoubleHeadsModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.value_head = nn.Linear(config.n_embd, 1, bias=False)
        self.value_head_target = nn.Linear(config.n_embd, 1, bias=False) # TODO: does not having a bias make sense?

        #self.multiple_choice_head = SequenceSummary(config)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    # def parallelize(self, device_map=None):
    #     self.device_map = (
    #         get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
    #         if device_map is None
    #         else device_map
    #     )
    #     assert_device_map(self.device_map, len(self.transformer.h))
    #     self.transformer.parallelize(self.device_map)
    #     self.lm_head = self.lm_head.to(self.transformer.first_device)
    #     self.multiple_choice_head = self.multiple_choice_head.to(self.transformer.first_device)
    #     self.model_parallel = True
    #
    # def deparallelize(self):
    #     self.transformer.deparallelize()
    #     self.transformer = self.transformer.to("cpu")
    #     self.lm_head = self.lm_head.to("cpu")
    #     self.multiple_choice_head = self.multiple_choice_head.to("cpu")
    #     self.model_parallel = False
    #     torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def train_value_head_only(self):
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.value_head.parameters():
            parameter.requires_grad = True

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        labels=None,
        mc_labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        rewards=None,
        gamma=0.95,
        **kwargs,
    ):
        r"""

        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        lm_loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        curr_value = self.value_head(hidden_states).squeeze(-1)
        curr_value = curr_value*attention_mask

        next_state_value = self.value_head_target(hidden_states).squeeze(-1)
        next_state_value = torch.roll(next_state_value, shifts=-1, dims=1)
        next_state_value = next_state_value*attention_mask

        td_loss = None
        if rewards is not None:

            reward_tensor = torch.zeros(curr_value.shape).to(self.device)
            for i in range(reward_tensor.shape[0]):
                last_tok_idx = int(torch.argmax(attention_mask[i,:]*torch.arange(attention_mask.shape[1]).to(self.device)))
                reward_tensor[i,last_tok_idx]=rewards.squeeze()[i]
                next_state_value[i,last_tok_idx]=0.

            target_value = reward_tensor + gamma*next_state_value

            td_loss = F.smooth_l1_loss(curr_value, target_value)

        output = (lm_logits, td_loss, curr_value) + transformer_outputs[1:]

        return ((lm_loss,) + output) if lm_loss is not None else output

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )
