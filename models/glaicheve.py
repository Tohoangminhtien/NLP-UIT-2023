import torch
from torch import nn

from builders.model_builder import META_ARCHITECTURE
from builders.text_embedding_builder import build_text_embedding
from models.modules.attentions import MultiHeadAttention
from models.modules.pos_embeddings import PositionalEmbedding
from models.modules.positionwise_feed_forward import PositionWiseFeedForward
from models.utils import generate_padding_mask, generate_sequential_mask
from data_utils.vocab import Vocab
from utils.instance import InstanceList

@META_ARCHITECTURE.register()
class GLAICHEVE(nn.Module):
    def __init__(self, config, vocab: Vocab) -> None:
        super().__init__()

        self.padding_idx = vocab.padding_idx
        self.max_sentence_length = vocab.max_sentence_length
        self.device = torch.device(config.DEVICE)

        self.embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)
        self.relu = nn.LeakyReLU()
        self.pe = PositionalEmbedding(
            d_model=config.D_MODEL,
            dropout=config.DROPOUT
        )

        self.self_attn = MultiHeadAttention(config.SELF_ATTENTION)
        self.cross_attn = MultiHeadAttention(config.CROSS_ATTENTION)
        self.pff = PositionWiseFeedForward(config.PPF)
        
        # verdict classifier
        self.verdict_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.verdict_classifier = nn.Linear(
            in_features=config.D_MODEL,
            out_features=3
        )
        
        self.evidence_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.evidence_selector = nn.Sequential(
            nn.Linear(
                in_features=config.D_MODEL,
                out_features=256
            ),
            nn.Linear(
                in_features=256,
                out_features=128
            ),
            nn.Linear(
                in_features=128,
                out_features=64
            ),
            nn.Linear(
                in_features=64,
                out_features=1
            )
        )
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, item: InstanceList) -> torch.Tensor:
        contexts = item.context
        claim = item.claim

        # apply the pre-trained word embedding
        contexts, _ = self.embedding(contexts)
        claim, _ = self.embedding(claim)

        # turn contexts into list of tensors
        contexts = [contexts[:, ith, :] for ith in range(contexts.shape[1])]

        # adding PE for contexts and project them into dimension of the model
        contexts = [self.pe(context) for context in contexts]
        contexts = [self.relu(context) for context in contexts]
        
        # adding PE for claim and project them into dimension of the model
        claim = self.pe(claim)
        claim = self.relu(claim)
        
        # apply self-attention on context and claim, respectively, then cross-attention for claim and every sentence in context
        claim_padding_mask = generate_padding_mask(claim, padding_idx=self.padding_idx).to(self.device)
        claim = self.self_attn(
            queries=claim,
            keys=claim,
            values=claim,
            attention_mask=claim_padding_mask.to(self.device)
        )
        claim = self.pff(claim)
        attended_contexts = []
        for context in contexts:
            context_padding_mask = generate_padding_mask(context, padding_idx=self.padding_idx).to(self.device)
            context = self.self_attn(
                queries=context, 
                keys=context, 
                values=context,
                attention_mask=context_padding_mask
            )
            cross_padding_mask = generate_sequential_mask(self.max_sentence_length).to(self.device)
            attended_context = self.cross_attn(
                    queries=context,
                    keys=claim,
                    values=claim,
                    attention_mask=cross_padding_mask
                )
            attended_context = self.pff(attended_context)
            attended_contexts.append(attended_context)

        attended_contexts = torch.concat([
            context.unsqueeze(1) for context in attended_contexts
        ], dim=1) # (bs, total_sentence, sentence_length, d_model)

        # verdict classification
        adapted_contexts = self.verdict_pooling(attended_contexts.permute((0, -1, 1, 2))).squeeze(-1).squeeze(-1) # (bs, d_model)
        verdict = self.verdict_classifier(adapted_contexts)
        verdict = self.softmax(verdict)

        # evidence extraction
        adapted_contexts = self.evidence_pooling(attended_contexts).squeeze(-1).squeeze(-1) # (bs, total_sentence)
        evidence = self.softmax(adapted_contexts)

        return {
            "verdict": verdict,
            "evidence": evidence
        }
