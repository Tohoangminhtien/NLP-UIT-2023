import torch
from torch.utils.data import DataLoader

from configs.utils import get_config
from builders.vocab_builder import build_vocab
from builders.dataset_builder import build_dataset
from builders.model_builder import build_model
from data_utils.utils import collate_fn

from tqdm import tqdm

device = torch.device("mps")

config = get_config("configs/GLAICHEVE.yaml")
vocab = build_vocab(config.DATASET.VOCAB)

train_dataset = build_dataset("annotations/refined-ise-dsc01-train.json", vocab, config.DATASET)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn
)

model = build_model(config.MODEL, vocab).to(device)

# for item in tqdm(train_dataset):
#     print(item.context.shape)
#     print(item.claim.shape)
#     print(item.verdict.shape)
#     print(item.evidence.shape)
#     raise

for item in tqdm(train_dataloader):
    gt_verdict = item.verdict
    gt_evidence = item.evidence
    
    context = item.context
    batch_sentences = item.context_sentences

    selected_evidence_ids = gt_evidence.argmax(dim=-1).tolist()
    selected_sentences = vocab.decode_evidence(selected_evidence_ids, batch_sentences)
    print(item.id)
    print(selected_sentences)
    raise
