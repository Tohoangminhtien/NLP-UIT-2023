import torch
from torch import nn
from torch.nn import functional as F

from builders.text_embedding_builder import META_TEXT_EMBEDDING
from builders.word_embedding_builder import build_word_embedding
from models.utils import generate_sequential_mask, generate_padding_mask

from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertPreTrainedModel,
)

from transformers.models.albert.modeling_albert import (
    AlbertConfig,
    AlbertEmbeddings,
    AlbertTransformer,
    AlbertPreTrainedModel
)

from transformers.models.roberta.modeling_roberta import (
    RobertaConfig,
    RobertaEmbeddings,
    RobertaEncoder,
    RobertaPreTrainedModel
)

from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Config,
    DebertaV2Embeddings,
    DebertaV2Encoder,
    DebertaV2PreTrainedModel
)

from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaConfig,
    XLMRobertaEmbeddings,
    XLMRobertaEncoder,
    XLMRobertaPreTrainedModel
)

from transformers import (
    BertTokenizer,
    AlbertTokenizer, 
    RobertaTokenizer,
    DebertaTokenizer,
    XLMTokenizer
)

from typing import List

@META_TEXT_EMBEDDING.register()
class VanillaEmbedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.padding_idx = vocab.padding_idx

        if config.WORD_EMBEDDING is None:
            self.components = nn.Embedding(len(vocab), config.D_MODEL, vocab.padding_idx)
        else:
            embedding_weights = build_word_embedding(config).vectors
            self.components = nn.Sequential(
                nn.Embedding.from_pretrained(embeddings=embedding_weights, freeze=True, padding_idx=vocab.padding_idx),
                nn.Linear(config.D_EMBEDDING, config.D_MODEL),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT)
            )

    def forward(self, tokens):
        padding_masks = generate_padding_mask(tokens, padding_idx=self.padding_idx).to(tokens.device)
        seq_len = tokens.shape[-1]
        sequential_masks = generate_sequential_mask(seq_len).to(tokens.device)

        features = self.components(tokens)

        return features, (padding_masks, sequential_masks)

@META_TEXT_EMBEDDING.register()
class RNNEmbedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.embedding = nn.Embedding(len(vocab), config.D_EMBEDDING, padding_idx=vocab.padding_idx)
        self.padding_idx = vocab.padding_idx
        if config.WORD_EMBEDDING is not None:
            embedding_weights = build_word_embedding(config).vectors
            self.embedding.from_pretrained(embedding_weights, freeze=True, padding_idx=vocab.padding_idx)
        self.proj = nn.Linear(config.D_EMBEDDING, config.D_MODEL)
        self.dropout = nn.Dropout(config.DROPOUT)

        self.lstm = nn.LSTM(input_size=config.D_MODEL, hidden_size=config.D_MODEL, batch_first=True)

    def forward(self, tokens):
        padding_masks = generate_padding_mask(tokens, padding_idx=self.padding_idx).to(tokens.device)
        seq_len = tokens.shape[-1]
        sequential_masks = generate_sequential_mask(seq_len).to(tokens.device)

        features = self.proj(self.embedding(tokens)) # (bs, seq_len, d_model)
        features = self.dropout(features)

        features, _ = self.lstm(features)

        return features, (padding_masks, sequential_masks)

@META_TEXT_EMBEDDING.register()
class HierarchicalFeaturesExtractor(nn.Module):
    def __init__(self, config, vocab) -> None:
        super().__init__()

        self.embedding = VanillaEmbedding(config, vocab)

        self.ngrams = config.N_GRAMS
        self.convs = nn.ModuleList()
        for ngram in self.ngrams:
            self.convs.append(
                nn.Conv1d(in_channels=config.D_MODEL, out_channels=config.D_MODEL, kernel_size=ngram)
            )

        self.reduce_features = nn.Linear(config.D_MODEL, config.D_MODEL)

    def forward(self, tokens: torch.Tensor):
        features, (padding_masks, sequential_masks) = self.embedding(tokens)

        ngrams_features = []
        for conv in self.convs:
            ngrams_features.append(conv(features.permute((0, -1, 1))).permute((0, -1, 1)))
        
        features_len = features.shape[-1]
        unigram_features = ngrams_features[0]
        # for each token in the unigram
        for ith in range(features_len):
            # for each n-gram, we ignore the unigram
            for ngram in range(1, max(self.ngrams)):
                # summing all possible n-gram tokens into the unigram
                for prev_ith in range(max(0, ith-ngram+1), min(ith+1, ngrams_features[ngram].shape[1])):
                    unigram_features[:, ith] += ngrams_features[ngram][:, prev_ith]

        return unigram_features, (padding_masks, sequential_masks)

class TextBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        encoder_inputs = self.embeddings(txt_inds)
        
        attention_mask = txt_mask
        head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(
            encoder_inputs, attention_mask, head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output
    
@META_TEXT_EMBEDDING.register()
class BertEmbedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.device = config.DEVICE

        bert_config = BertConfig(
            hidden_size=config.HIDDEN_SIZE,
            num_hidden_layers=config.NUM_HIDDEN_LAYERS,
            num_attention_heads=config.NUM_ATTENTION_HEADS
        )

        self.tokenizer = BertTokenizer.from_pretrained(config.PRETRAINED_NAME)
        self.embedding = TextBert(bert_config)
        if config.LOAD_PRETRAINED:
            self.embedding = self.embedding.from_pretrained(config.PRETRAINED_NAME)
        if config.FREEZE_WEIGHTS:
            # freeze all parameters of pretrained model
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.proj = nn.Linear(config.HIDDEN_SIZE, config.D_MODEL)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, questions: List[str]):
        inputs = self.tokenizer(questions, return_tensors="pt", padding=True).input_ids.to(self.device)
        padding_mask = generate_padding_mask(inputs, padding_idx=self.tokenizer.pad_token_id)
        features = self.embedding(inputs, padding_mask)

        out = self.proj(features)
        out = self.dropout(self.gelu(out))

        return out, padding_mask
    
class TextAlbert(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.embeddings = AlbertEmbeddings(config)
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.encoder = AlbertTransformer(config)
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        embedded_inputs = self.embeddings(txt_inds)
        encoder_inputs = self.embedding_hidden_mapping_in(embedded_inputs)
        
        attention_mask = txt_mask
        head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(
            encoder_inputs, attention_mask, head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output
    
@META_TEXT_EMBEDDING.register()
class AlbertEmbedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.device = config.DEVICE

        albert_config = AlbertConfig(
            hidden_size=config.HIDDEN_SIZE,
            num_hidden_layers=config.NUM_HIDDEN_LAYERS,
            num_attention_heads=config.NUM_ATTENTION_HEADS
        )

        self.tokenizer = AlbertTokenizer.from_pretrained(config.PRETRAINED_NAME)
        self.embedding = TextAlbert(albert_config)
        if config.LOAD_PRETRAINED:
            self.embedding = self.embedding.from_pretrained(config.PRETRAINED_NAME)
        if config.FREEZE_WEIGHTS:
            # freeze all parameters of pretrained model
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.proj = nn.Linear(config.HIDDEN_SIZE, config.D_MODEL)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, questions: List[str]):
        inputs = self.tokenizer(questions, return_tensors="pt", padding=True).input_ids.to(self.device)
        padding_mask = generate_padding_mask(inputs, padding_idx=self.tokenizer.pad_token_id)
        features = self.embedding(inputs, padding_mask)

        out = self.proj(features)
        out = self.dropout(self.gelu(out))

        return out, padding_mask

class TextRoberta(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        encoder_inputs = self.embeddings(txt_inds)
        
        attention_mask = txt_mask
        head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(
            encoder_inputs, attention_mask, head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output
    
@META_TEXT_EMBEDDING.register()
class RobertaEmbedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.device = config.DEVICE

        roberta_config = RobertaConfig(
            hidden_size=config.HIDDEN_SIZE,
            num_hidden_layers=config.NUM_HIDDEN_LAYERS,
            num_attention_heads=config.NUM_ATTENTION_HEADS
        )

        self.tokenizer = RobertaTokenizer.from_pretrained(config.PRETRAINED_NAME)
        self.embedding = TextRoberta(roberta_config)
        if config.LOAD_PRETRAINED:
            self.embedding = self.embedding.from_pretrained(config.PRETRAINED_NAME)
        if config.FREEZE_WEIGHTS:
            # freeze all parameters of pretrained model
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.proj = nn.Linear(config.HIDDEN_SIZE, config.D_MODEL)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, questions: List[str]):
        inputs = self.tokenizer(questions, return_tensors="pt", padding=True).input_ids.to(self.device)
        padding_mask = generate_padding_mask(inputs, padding_idx=self.tokenizer.pad_token_id)
        features = self.embedding(inputs, padding_mask)

        out = self.proj(features)
        out = self.dropout(self.gelu(out))

        return out, padding_mask
    
class TextDeberta_v2(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DebertaV2Encoder(config)
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        encoder_inputs = self.embeddings(txt_inds)
        
        attention_mask = txt_mask
        head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(
            encoder_inputs, attention_mask, head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output
    
@META_TEXT_EMBEDDING.register()
class DebertaEmbedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.device = config.DEVICE

        deberta_config = DebertaV2Config(
            hidden_size=config.HIDDEN_SIZE,
            num_hidden_layers=config.NUM_HIDDEN_LAYERS,
            num_attention_heads=config.NUM_ATTENTION_HEADS
        )

        self.tokenizer = DebertaTokenizer.from_pretrained(config.PRETRAINED_NAME)
        self.embedding = TextDeberta_v2(deberta_config)
        if config.LOAD_PRETRAINED:
            self.embedding = self.embedding.from_pretrained(config.PRETRAINED_NAME)
        if config.FREEZE_WEIGHTS:
            # freeze all parameters of pretrained model
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.proj = nn.Linear(config.HIDDEN_SIZE, config.D_MODEL)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, questions: List[str]):
        inputs = self.tokenizer(questions, return_tensors="pt", padding=True).input_ids.to(self.device)
        padding_mask = generate_padding_mask(inputs, padding_idx=self.tokenizer.pad_token_id)
        features = self.embedding(inputs, padding_mask)

        out = self.proj(features)
        out = self.dropout(self.gelu(out))

        return out, padding_mask
    
class TextXLM(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.embeddings = XLMRobertaEmbeddings(config)
        self.encoder = XLMRobertaEncoder(config)
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        encoder_inputs = self.embeddings(txt_inds)
        
        attention_mask = txt_mask
        head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(
            encoder_inputs, attention_mask, head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output
    
@META_TEXT_EMBEDDING.register()
class XLMRobertaEmbedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.device = config.DEVICE

        xlm_config = XLMRobertaConfig(
            hidden_size=config.HIDDEN_SIZE,
            num_hidden_layers=config.NUM_HIDDEN_LAYERS,
            num_attention_heads=config.NUM_ATTENTION_HEADS
        )

        self.tokenizer = XLMTokenizer.from_pretrained(config.PRETRAINED_NAME)
        self.embedding = TextXLM(xlm_config)
        if config.LOAD_PRETRAINED:
            self.embedding = self.embedding.from_pretrained(config.PRETRAINED_NAME)
        if config.FREEZE_WEIGHTS:
            # freeze all parameters of pretrained model
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.proj = nn.Linear(config.HIDDEN_SIZE, config.D_MODEL)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, questions: List[str]):
        inputs = self.tokenizer(questions, return_tensors="pt", padding=True).input_ids.to(self.device)
        padding_mask = generate_padding_mask(inputs, padding_idx=self.tokenizer.pad_token_id)
        features = self.embedding(inputs, padding_mask)

        out = self.proj(features)
        out = self.dropout(self.gelu(out))

        return out, padding_mask