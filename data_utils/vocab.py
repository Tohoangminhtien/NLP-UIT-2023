import torch

from builders.word_embedding_builder import build_word_embedding
from data_utils.utils import preprocess_sentence, segment_context, unk_init
from builders.vocab_builder import META_VOCAB

from collections import Counter
import json
from typing import List, Union

@META_VOCAB.register
class Vocab(object):
    """
        Defines a vocabulary object that will be used to numericalize a field.
    """
    def __init__(self, config):

        self.tokenizer = config.TOKENIZER

        self.padding_token = config.PAD_TOKEN
        self.bos_token = config.BOS_TOKEN
        self.eos_token = config.EOS_TOKEN
        self.unk_token = config.UNK_TOKEN

        self.make_vocab([
            config.JSON_PATH.TRAIN,
            config.JSON_PATH.DEV,
            config.JSON_PATH.TEST
        ])
        counter = self.freqs.copy()
    
        min_freq = max(config.MIN_FREQ, 1)

        specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token]
        itos = specials
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq:
                break
            itos.append(word)

        self.itos = {i: tok for i, tok in enumerate(itos)}
        self.stoi = {tok: i for i, tok in enumerate(itos)}

        self.specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token]

        self.padding_idx = self.stoi[self.padding_token]
        self.bos_idx = self.stoi[self.bos_token]
        self.eos_idx = self.stoi[self.eos_token]
        self.unk_idx = self.stoi[self.unk_token]

        self.word_embeddings = None
        if config.WORD_EMBEDDING is not None:
            self.load_word_embeddings(build_word_embedding(config))

    def make_vocab(self, json_dirs):
        self.freqs = Counter()
        self.max_sentence_length = 0
        self.max_context_sentences = 0
        for json_dir in json_dirs:
            if json_dir is None:
                continue
            json_data = json.load(open(json_dir))
            for id in json_data:
                item = json_data[id]
                context = item["context"]
                context_sentences = segment_context(context)
                if len(context_sentences) > self.max_context_sentences:
                    self.max_context_sentences = len(context_sentences)
                context_sentences = [preprocess_sentence(sentence) for sentence in context_sentences]
                claim = item["claim"]
                claim = preprocess_sentence(claim)
                for sentence in context_sentences:
                    self.freqs.update(sentence)
                    if self.max_sentence_length < len(sentence) + 2: # extend the length for <bos> and <eos>
                        self.max_sentence_length = len(sentence) + 2
                self.freqs.update(claim)
                if self.max_sentence_length < len(claim) + 2:
                        self.max_sentence_length = len(claim) + 2

    def encode_sentence(self, sentence: str, tag: str) -> torch.Tensor:
        """ 
            sentence: list of string tokens
        """
        if tag == "SENTENCE":
            sentence = preprocess_sentence(sentence)
            vec = torch.ones(self.max_sentence_length).long() * self.padding_idx
            for i, token in enumerate([self.bos_token] + sentence + [self.eos_token]):
                vec[i] = self.stoi[token] if token in self.stoi else self.unk_idx

            return vec
        
        if tag == "PARAGRAPH":
            sentences = segment_context(sentence)
            processed_sentences = [preprocess_sentence(sentence) for sentence in sentences]
            vec = torch.ones((self.max_context_sentences, self.max_sentence_length)).long() * self.padding_idx
            for i, sentence in enumerate(processed_sentences):
                for j, token in enumerate([self.bos_token] + sentence + [self.eos_token]):
                    vec[i][j] = self.stoi[token] if token in self.stoi else self.unk_idx
            
            return vec, sentences
        
        raise "tag is not valid"
    
    def encode_verdict(self, verdict: str) -> torch.Tensor:
        if verdict == "SUPPORTED":
            return torch.tensor([1, 0, 0])
        if verdict == "NEI":
            return torch.tensor([0, 1, 0])
        if verdict == "REFUTED":
            return torch.tensor([0, 0, 1])
        
    def encode_evidence(self, context: str, evidence: Union[str, None]) -> torch.Tensor:
        context_sentences = segment_context(context)
        context_sentences = [preprocess_sentence(sentence) for sentence in context_sentences]

        if evidence is not None:
            processed_evidence = preprocess_sentence(evidence)
            assert processed_evidence in context_sentences
            evidence_index = context_sentences.index(processed_evidence)
        else:
            evidence_index = self.max_context_sentences

        vec = torch.zeros((self.max_context_sentences+1, )).long()
        vec[evidence_index] = 1
        
        return vec

    def decode_verdict(self, verdict_vec: torch.Tensor) -> List[str]:
        '''
            verdict_vec: (bs, 3)
        '''
        verdict_map = {
            0: "SUPPORTED",
            1: "NEI",
            2: "REFUTED"
        }
        verdicts = verdict_vec.argmax(dim=-1).long().tolist()
        verdicts = [verdict_map[verdict] for verdict in verdicts]

        return verdicts
    
    def decode_evidence(self, selected_sentence_ids: List[torch.Tensor], batch_sentences: List[List[str]]) -> List[str]:
        evidences = []
        for selected_sentence_id, sentences in zip(selected_sentence_ids, batch_sentences):
            if selected_sentence_id < self.max_context_sentences:
                evidences.append(sentences[selected_sentence_id])
            else:
                evidences.append("None")

        return evidences

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.word_embeddings != other.word_embeddings:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def extend(self, v, sort=False):
        words = sorted(v.itos.values()) if sort else v.itos.values()
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1

    def load_word_embeddings(self, word_embeddings):
        if not isinstance(word_embeddings, list):
            word_embeddings = [word_embeddings]

        tot_dim = sum(embedding.dim for embedding in word_embeddings)
        self.word_embeddings = torch.Tensor(len(self), tot_dim)
        for i, token in self.itos.items():
            start_dim = 0
            for v in word_embeddings:
                end_dim = start_dim + v.dim
                self.word_embeddings[i][start_dim:end_dim] = v[token.strip()]
                start_dim = end_dim
            assert(start_dim == tot_dim)

    def set_vectors(self, stoi, word_embeddings, dim):
        """
        Set the word_embeddings for the Vocab instance from a collection of Tensors.
        Arguments:
            stoi: A dictionary of string to the index of the associated vector
                in the `word_embeddings` input argument.
            word_embeddings: An indexed iterable (or other structure supporting __getitem__) that
                given an input index, returns a FloatTensor representing the vector
                for the token associated with the index. For example,
                vector[stoi["string"]] should return the vector for "string".
            dim: The dimensionality of the word_embeddings.
        """
        self.word_embeddings = torch.Tensor(len(self), dim)
        for i, token in self.itos.items():
            we_index = stoi.get(token, None)
            if we_index is not None:
                self.word_embeddings[i] = word_embeddings[we_index]
            else:
                self.word_embeddings[i] = unk_init(self.word_embeddings[i])
