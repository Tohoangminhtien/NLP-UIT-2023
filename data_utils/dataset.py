from torch.utils.data import Dataset

from utils.instance import Instance
from data_utils.vocab import Vocab
from builders.dataset_builder import META_DATASET

import json

@META_DATASET.register
class IcheveDataset(Dataset):
    def __init__(self, annotation_path: str, vocab: Vocab, config) -> None:
        super(Dataset, self).__init__()

        self.vocab = vocab
        self.config = config

        self.__annotations = json.load(open(annotation_path))
        self.__ids = [id for id in self.__annotations]

    def __len__(self) -> int:
        return len(self.__ids)

    def __getitem__(self, index: int) -> Instance:
        id = self.__ids[index]
        item = self.__annotations[id]

        context = item["context"]
        claim = item["claim"]
        verdict = item["verdict"]
        evidence = item["evidence"]

        context_tensor, context_sentences = self.vocab.encode_sentence(context, "PARAGRAPH")
        print(len(context_sentences))
        claim_tensor = self.vocab.encode_sentence(claim, "SENTENCE")
        verdict_tensor = self.vocab.encode_verdict(verdict)
        evidence_tensor = self.vocab.encode_evidence(context, evidence)

        return Instance(
            id = self.__ids[index],
            context_sentences = context_sentences,
            context = context_tensor,
            claim = claim_tensor,
            verdict = verdict_tensor,
            evidence = evidence_tensor
        )
