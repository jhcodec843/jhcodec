import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2BertModel, SeamlessM4TFeatureExtractor
from transformers import AutoConfig
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import Wav2Vec2BertEncoderLayer, Wav2Vec2BertRelPositionalEmbedding, Wav2Vec2BertRotaryPositionalEmbedding
import logging
logging.basicConfig(level=logging.INFO)


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2_bert/modular_wav2vec2_bert.py
# adopted from the huggingface transformers
# revised the forward function for efficient inference until the target layer
class Wav2Vec2BertEncoder(nn.Module):
    def __init__(self, config, target_layer=17):
        super().__init__()
        self.config = config
        self.target_layer = target_layer

        if config.position_embeddings_type == "relative":
            self.embed_positions = Wav2Vec2BertRelPositionalEmbedding(config)
        elif config.position_embeddings_type == "rotary":
            self.embed_positions = Wav2Vec2BertRotaryPositionalEmbedding(config)
        else:
            self.embed_positions = None

        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([Wav2Vec2BertEncoderLayer(config) for _ in range(target_layer)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
    ):
        conv_attention_mask = attention_mask
        if attention_mask is not None:
            # make sure padded tokens output 0
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        hidden_states = self.dropout(hidden_states)

        if self.embed_positions is not None:
            relative_position_embeddings = self.embed_positions(hidden_states)
        else:
            relative_position_embeddings = None

        synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

        for i, layer in enumerate(self.layers[:self.target_layer]):
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = self.training and dropout_probability < self.config.layerdrop
            if not skip_the_layer or synced_gpus:
                # under fsdp or deepspeed zero3 all gpus must run in sync
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    relative_position_embeddings=relative_position_embeddings,
                    output_attentions=output_attentions,
                    conv_attention_mask=conv_attention_mask,
                )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)
        # Only return the 17th layer output as last_hidden_state
        if not return_dict:
            return (hidden_states,)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=None,
        )


class W2V2Bert2FeatureWrapper():

    def __init__(self,
                 model_ckpt="facebook/w2v-bert-2.0",
                 device=None,
                 target_layer=17,
                 **kwargs,
                ):
        super().__init__()
        self.model = Wav2Vec2BertModel.from_pretrained(model_ckpt)

        config = self.model.config
        self.encoder = Wav2Vec2BertEncoder(config, target_layer=target_layer)
        # Copy model's state_dict to encoder for existing values
        model_state_dict = self.model.encoder.state_dict()
        encoder_state_dict = self.encoder.state_dict()
        # Only load matching keys
        matched_state_dict = {k: v for k, v in model_state_dict.items() if k in encoder_state_dict and v.shape == encoder_state_dict[k].shape}
        self.encoder.load_state_dict(matched_state_dict, strict=False)
        self.model.encoder = self.encoder

        self.processor = SeamlessM4TFeatureExtractor.from_pretrained(model_ckpt)
        if device == None:
            device = "cpu"
        self.model.to(device)
        #self.encoder.to(device)
        #self.encoder.eval()
        self.model.eval()
        #self.model = torch.compile(self.model, dynamic=True)
        
        self.device = device


    @torch.no_grad()
    def __call__(self, input_features, attention_mask):
        """
        Process single wav file or a list of wav files through the model
        
        Args:
            wavs: tensor of shape [B, T]
            original_lengths: original lengths of the wavs [B]
        Returns:
            For single file: Dictionary with segments and segment_features
            For multiple files: List of dictionaries, each with segments and segment_features
        """
        #wavs_cpu = wavs.cpu().numpy()
        #inputs = self.processor(wavs_cpu, sampling_rate=16000, return_tensors="pt")
        #input_features = inputs["input_features"].to(self.device)
        #attention_mask = inputs["attention_mask"].to(self.device)

        output = self.model(input_features=input_features, attention_mask=attention_mask, output_hidden_states=False)
        # The output hidden state from Wav2Vec2BertModel is in shape (batch_size, sequence_length, hidden_size): [B, T, C]
        hidden_state = output.last_hidden_state  # shape: [B, T, C]
        #assert hidden_state.isnan().any() == False, "hidden_state is nan"
        if hidden_state.isnan().any():
            logging.info(f"hidden_state is nan")
            hidden_state = torch.where(hidden_state.isnan(), -32, hidden_state)
        hidden_state = hidden_state.to(torch.float32)
        hidden_state = hidden_state
        return hidden_state

    @torch.no_grad()
    def full_forward(self, wavs):
        wavs_cpu = wavs.cpu().numpy()
        inputs = self.processor(wavs_cpu, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        return self.__call__(input_features, attention_mask)
  

if __name__ == "__main__":
    model = W2V2Bert2FeatureWrapper(device= "cuda")
    wavs = torch.randn(1, 16000)
    inputs = model.processor(wavs, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    hidden_state = model(input_features=input_features, attention_mask=attention_mask, output_hidden_states=True).hidden_states[17]
    print(hidden_state.shape)