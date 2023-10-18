import torch
from torch import nn

from models.modules.positionwise_feed_forward import PositionWiseFeedForward
from models.modules.attentions import MultiHeadAttention
from models.modules.pos_embeddings import SinusoidPositionalEmbedding
from builders.encoder_builder import META_ENCODER

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.mhatt = MultiHeadAttention(config)
        self.pwff = PositionWiseFeedForward(config)

    def forward(self, queries, keys, values, attention_mask, **kwargs):
        att = self.mhatt(queries=queries, keys=keys, values=values, attention_mask=attention_mask, **kwargs)
        ff = self.pwff(att)

        return ff

@META_ENCODER.register()
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        self.pos_embedding = SinusoidPositionalEmbedding(config.D_MODEL)
        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.d_model = config.D_MODEL
        self.layers = nn.ModuleList([EncoderLayer(config.SELF_ATTENTION) for _ in range(config.LAYERS)])

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor):
        out = self.layer_norm(features) + self.pos_embedding(features)
        for layer in self.layers:
            out = layer(queries=out, keys=out, values=out, attention_mask=padding_mask)

        return out

