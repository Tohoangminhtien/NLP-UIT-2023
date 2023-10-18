import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from configs.utils import get_config
from builders.vocab_builder import build_vocab
from builders.dataset_builder import build_dataset
from builders.model_builder import build_model
from data_utils.utils import collate_fn
from evaluation.evaluate import strict_accuracy

from tqdm import tqdm
import os

config = get_config("configs/baseline.yaml")
device = torch.device(config.DEVICE)

vocab = build_vocab(config.DATASET.VOCAB)

train_dataset = build_dataset(config.JSON_PATH.TRAIN, vocab, config.DATASET)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=config.DATASET.BATCH_SIZE,
    shuffle=True,
    num_workers=config.DATASET.WORKERS,
    collate_fn=collate_fn
)

dev_dataset = build_dataset(config.JSON_PATH.DEV, vocab, config.DATASET)
dev_dataloader = DataLoader(
    dataset=dev_dataset,
    batch_size=1,
    num_workers=config.DATASET.WORKERS,
    shuffle=True,
    collate_fn=collate_fn
)

model = build_model(config.MODEL, vocab).to(device)
optimizer = Adam(model.parameters(), lr=config.LR)
loss_fn = nn.CrossEntropyLoss().to(device)

epoch = 0
patient = 0
best_strict_acc = 0
best_scores = {}
while True:
    epoch +=1

    with tqdm(train_dataloader, desc=f"Epoch {epoch} - Training") as pb:
        for item in pb:
            item = item.to(device)
            # forward
            output = model(item)
            verdict = output["verdict"]
            evidence = output["evidence"]
            # backward
            gt_verdict = item.verdict
            gt_evidence = item.evidence
            loss = 0.5*loss_fn(verdict, gt_verdict) + 0.5*loss_fn(evidence, gt_evidence)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pb.set_postfix({"Loss": loss.item()})

    with tqdm(dev_dataloader, desc=f"Epoch {epoch} - Evaluating") as pb:
        gts = {}
        preds = {}
        for item in pb:
            item = item.to(device)
            # forward
            with torch.no_grad():
                output = model(item)
                context = item.context
                batch_sentences = item.context_sentences

                # decoding predicted verdict
                verdict = output["verdict"]
                verdict = vocab.decode_verdict(verdict)
                # decoding ground truth verdict
                gt_verdict = item.verdict
                gt_verdict = vocab.decode_verdict(gt_verdict)

                # decoding ground truth evidence
                gt_evidence = item.evidence
                selected_gt_evidence_ids = gt_evidence.argmax(dim=-1).tolist()
                selected_gt_sentences = vocab.decode_evidence(selected_gt_evidence_ids, batch_sentences)
                # decoding predicted evidence
                evidence = output["evidence"]
                selected_evidence_ids = evidence.argmax(dim=-1).tolist()
                selected_sentences = vocab.decode_evidence(selected_evidence_ids, batch_sentences)

                preds[item.id[0]] = {
                    "verdict": verdict[0],
                    "evidence": evidence[0]
                }
                gts[item.id[0]] = {
                    "verdict": gt_verdict[0],
                    "evidence": gt_evidence[0]
                }

    scores = strict_accuracy(gts, preds)
    if best_strict_acc < scores["strict_acc"]:
        best_strict_acc = scores["strict_acc"]
        best_scores = scores
        # save the best model
        saving_dir = os.path.join(config.TRAINING.CHECKPOINT_PATH, config.MODEL.NAME)
        if not os.path.isdir(saving_dir):
            os.mkdir(saving_dir)
        torch.save(model.parameters(), saving_dir)
        continue

    patient += 1
    if patient > config.TRAINING.PATIENT:
        break
