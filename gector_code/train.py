"""Pure PyTorch training script for GECToR"""
import argparse
import json
import os
import random

import numpy as np
import torch
from transformers import AutoTokenizer

from gector.datareader import Seq2LabelsDatasetReader
from gector.seq2labels_model import Seq2Labels
from gector.trainer import Trainer
from utils.helpers import get_weights_name


# Default vocabulary files location
DEFAULT_VOCAB_PATH = os.path.join(
    os.path.dirname(__file__), "data", "output_vocabulary"
)


def fix_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_vocab(vocab_path: str):
    """Load vocabulary from files"""
    vocab = {"labels": {}, "d_tags": {}}

    labels_file = os.path.join(vocab_path, "labels.txt")
    d_tags_file = os.path.join(vocab_path, "d_tags.txt")

    with open(labels_file, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            token = line.strip()
            if token:
                vocab["labels"][token] = idx

    with open(d_tags_file, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            token = line.strip()
            if token:
                vocab["d_tags"][token] = idx

    # Add default tokens if missing
    for namespace in ["labels", "d_tags"]:
        if "@@PADDING@@" not in vocab[namespace]:
            vocab[namespace]["@@PADDING@@"] = len(vocab[namespace])
        if "@@UNKNOWN@@" not in vocab[namespace]:
            vocab[namespace]["@@UNKNOWN@@"] = len(vocab[namespace])

    print(f"Vocab loaded: {len(vocab['labels'])} labels, {len(vocab['d_tags'])} d_tags")
    return vocab


def get_data_reader(model_name, max_len, skip_correct=False, skip_complex=0,
                    test_mode=False, tag_strategy="keep_one",
                    broken_dot_strategy="keep", lowercase_tokens=True,
                    max_pieces_per_token=3, tn_prob=0, tp_prob=1,
                    special_tokens_fix=0):
    """Create data reader"""
    reader = Seq2LabelsDatasetReader(
        max_len=max_len,
        skip_correct=skip_correct,
        skip_complex=skip_complex,
        test_mode=test_mode,
        tag_strategy=tag_strategy,
        broken_dot_strategy=broken_dot_strategy,
        tn_prob=tn_prob,
        tp_prob=tp_prob
    )
    return reader


def get_model(model_name, vocab, tune_bert=True,
              predictor_dropout=0.0,
              label_smoothing=0.0,
              confidence=0.0,
              special_tokens_fix=0):
    """Create GECToR model"""
    num_labels = len(vocab["labels"])
    num_detect = len(vocab["d_tags"])

    model = Seq2Labels(
        encoder_name=model_name,
        num_labels_classes=num_labels,
        num_detect_classes=num_detect,
        predictor_dropout=predictor_dropout,
        label_smoothing=label_smoothing,
        confidence=confidence,
        tune_bert=bool(tune_bert),
        special_tokens_fix=special_tokens_fix
    )

    # Set incorr_index from vocab
    incorr_idx = vocab["d_tags"].get("INCORRECT", 1)
    model.incorr_index = incorr_idx

    return model


def main(args):
    fix_seed()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir, exist_ok=True)

    # Get HuggingFace model name
    weights_name = get_weights_name(args.transformer_model, args.lowercase_tokens)
    print(f"Using encoder: {weights_name}")

    # Load vocabulary
    vocab_path = args.vocab_path if args.vocab_path else DEFAULT_VOCAB_PATH
    vocab = load_vocab(vocab_path)

    # Save vocab to model dir
    import shutil
    if not os.path.exists(os.path.join(args.model_dir, "vocabulary")):
        shutil.copytree(vocab_path, os.path.join(args.model_dir, "vocabulary"))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(weights_name)
    if args.special_tokens_fix:
        tokenizer.add_tokens(['$START'])

    # Read data
    reader = get_data_reader(
        weights_name,
        args.max_len,
        skip_correct=bool(args.skip_correct),
        skip_complex=args.skip_complex,
        test_mode=False,
        tag_strategy=args.tag_strategy,
        lowercase_tokens=args.lowercase_tokens,
        max_pieces_per_token=args.pieces_per_token,
        tn_prob=args.tn_prob,
        tp_prob=args.tp_prob,
        special_tokens_fix=args.special_tokens_fix
    )

    print(f"Reading training data from {args.train_set}...")
    train_data = reader.read(args.train_set)
    print(f"Reading validation data from {args.dev_set}...")
    dev_data = reader.read(args.dev_set)
    print(f"Train: {len(train_data)} instances, Dev: {len(dev_data)} instances")

    # Build model
    model = get_model(
        weights_name,
        vocab,
        tune_bert=args.tune_bert,
        predictor_dropout=args.predictor_dropout,
        label_smoothing=args.label_smoothing,
        special_tokens_fix=args.special_tokens_fix
    )

    # Load pretrained weights if specified
    if args.pretrain and args.pretrain_folder:
        pretrain_path = os.path.join(args.pretrain_folder, f"{args.pretrain}.th")
        if os.path.exists(pretrain_path):
            print(f"Loading pretrained weights from {pretrain_path}")
            state_dict = torch.load(pretrain_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print("Pretrained weights loaded!")
        else:
            print(f"Warning: pretrain file not found at {pretrain_path}")

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {device}")

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10
    )

    # Calculate updates per epoch
    updates_per_epoch = args.updates_per_epoch if args.updates_per_epoch else None

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_data,
        validation_dataset=dev_data,
        vocab=vocab,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_epochs=args.n_epoch,
        patience=args.patience,
        serialization_dir=args.model_dir,
        cuda_device=cuda_device,
        accumulated_batch_count=args.accumulation_size,
        cold_step_count=args.cold_steps_count,
        cold_lr=args.cold_lr,
        updates_per_epoch=updates_per_epoch,
        shuffle=False
    )

    print("Starting training...")
    metrics = trainer.train()

    # Save final model
    out_model = os.path.join(args.model_dir, "model.th")
    torch.save(model.state_dict(), out_model)
    print(f"Model saved to {out_model}")

    # Save metrics
    metrics_path = os.path.join(args.model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set', required=True)
    parser.add_argument('--dev_set', required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--vocab_path', default='')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--target_vocab_size', type=int, default=1000)
    parser.add_argument('--n_epoch', type=int, default=20)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--skip_correct', type=int, default=1)
    parser.add_argument('--skip_complex', type=int,
                        choices=[0, 1, 2, 3, 4, 5], default=0)
    parser.add_argument('--tune_bert', type=int, default=1)
    parser.add_argument('--tag_strategy',
                        choices=['keep_one', 'merge_all'], default='keep_one')
    parser.add_argument('--accumulation_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--cold_steps_count', type=int, default=4)
    parser.add_argument('--cold_lr', type=float, default=1e-3)
    parser.add_argument('--predictor_dropout', type=float, default=0.0)
    parser.add_argument('--lowercase_tokens', type=int, default=0)
    parser.add_argument('--pieces_per_token', type=int, default=5)
    parser.add_argument('--cuda_verbose_steps', default=None)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--tn_prob', type=float, default=0)
    parser.add_argument('--tp_prob', type=float, default=1)
    parser.add_argument('--updates_per_epoch', type=int, default=0)
    parser.add_argument('--pretrain_folder', default='')
    parser.add_argument('--pretrain', default='')
    parser.add_argument('--transformer_model',
                        choices=['bert', 'distilbert', 'gpt2', 'roberta',
                                 'transformerxl', 'xlnet', 'albert',
                                 'tinybert', 'mobilebert'],
                        default='roberta')
    parser.add_argument('--special_tokens_fix', type=int, default=1)

    args = parser.parse_args()
    main(args)