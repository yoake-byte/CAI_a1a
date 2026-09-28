"""Course-provided skeleton for training Part A (CRF) and Part B (BiLSTM) models.

In this training script, CLI parsing, data loading, and model construction are wired up for you.
The only one thing left to implement is `train_loop()` — actual training loop,
for both Part A CRF and Part B BiLSTM. Look for the `# TODO: implement` block below.

Usage:
    python train.py --model crf    --train-fraction 1.00 --checkpoint-dir checkpoints/crf_1.00
    python train.py --model bilstm --train-fraction 0.25 --checkpoint-dir checkpoints/bilstm_0.25 \
        --tensorboard-logdir runs/bilstm_0.25
"""

from __future__ import annotations
import argparse
from csv import writer
import os
import random
from collections import Counter
from typing import Any
import numpy as np
import torch
import joblib
import data_loader
from torch.utils.tensorboard import SummaryWriter

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a training run."""
    parser = argparse.ArgumentParser(
        description="Train a CRF or BiLSTM slot-filling model on ATIS."
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["crf", "bilstm"],
        required=True,
        help="Which model to train.",
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        required=True,
        help="Fraction of training set to use, e.g. 0.05 / 0.10 / 0.25 / 1.00.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory to save the trained model to.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    parser.add_argument(
        "--tensorboard-logdir",
        type=str,
        default=None,
        help="TensorBoard log directory. Only used when --model BiLSTM; "
        "ignored for --model crf (CRF training doesn't need TensorBoard or a GPU).",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch — CPU + CUDA — for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(fraction: float, seed: int) -> tuple[Any, Any, Any, Any]:
    """Load the ATIS train/dev/test splits, subsampling the train split.

    Returns:
        (`train_data`, `dev_data`, `test_data`, `vocab`)

        `train_data`/`dev_data`/`test_data` are `list[data_loader.Example]`.
        `vocab` is the dict returned by `build_vocab()` (see below),
        augmented with `tagset_size` (via `data_loader.slot_label_set()`) —
        needed to construct `BiLSTMModel` in `build_model()`.
    """
    import data_loader

    all_splits = data_loader.load_atis()
    train_data = data_loader.subsample(
        all_splits["train"], fraction=fraction, seed=seed
    )

    dev_data = all_splits["dev"]
    test_data = all_splits["test"]

    train_sentences = [example.tokens for example in train_data]
    vocab = build_vocab(train_sentences)
    vocab["tagset_size"] = len(data_loader.slot_label_set(train_data))

    return train_data, dev_data, test_data, vocab


# Built here rather than in data_loader.py to avoid touching a
# teammate-owned file; if data_loader.py later grows its own vocab
# utility, this should be removed in favor of that to avoid two
# divergent implementations.
def build_vocab(train_sentences: list[list[str]]) -> dict[str, Any]:
    """Build a token vocabulary from tokenized training sentences.

    Fully implemented — this is infrastructure, not the student-fill part
    of the assignment.

    Reserves index 0 for "<PAD>" and index 1 for "<UNK>". Tokens are then
    ordered by descending frequency, ties broken alphabetically. This
    determinism is required: `token_to_id` must be identical across
    machines/Python versions/runs, or it risks breaking the ±0.3 F1
    reproducibility tolerance in the grading rubric. Do NOT rely on dict or
    set iteration order for this — the explicit `sorted()` key below is
    what makes it deterministic.

    Args:
        train_sentences: Tokenized training sentences.

    Returns:
        A dict with `token_to_id` (Dict[str, int]), `id_to_token`
        (Dict[int, str]), and `vocab_size` (int).
    """
    counts = Counter(token for sentence in train_sentences for token in sentence)
    ordered_tokens = sorted(counts, key=lambda token: (-counts[token], token))

    token_to_id = {"<PAD>": 0, "<UNK>": 1}
    for token in ordered_tokens:
        token_to_id[token] = len(token_to_id)
    id_to_token = {index: token for token, index in token_to_id.items()}

    return {
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "vocab_size": len(token_to_id),
    }


def build_model(
    model_type: str,
    vocab_size: int | None = None,
    tagset_size: int | None = None,
) -> Any:
    """Construct an untrained model instance for the given model type.

    `vocab_size`/`tagset_size` (from `load_data()`'s `vocab`) are required
    for `model_type == "bilstm"` and ignored for `model_type == "crf"` —
    `CRFModel` works on raw token features, not a fixed vocabulary, so
    there's nothing for it to size itself against. This is intentional,
    not a bug, same as `CRFModel.predict()`'s unused `vocab` parameter.
    """
    if model_type == "crf":
        from models.crf import CRFModel

        return CRFModel()

    if model_type == "bilstm":
        from models.bilstm import BiLSTMModel

        if vocab_size is None or tagset_size is None:
            raise ValueError(
                "vocab_size and tagset_size are required for model_type='bilstm'"
            )

        return BiLSTMModel(vocab_size=vocab_size, tagset_size=tagset_size)

    raise ValueError(f"Unknown model type: {model_type!r}")


def extract_spans(tags: list[str]) -> set[tuple[str, int, int]]:
    """Extracts (entity_type, start_idx, end_idx) from BIO tags."""
    spans = set()
    current_span = None
    
    for i, tag in enumerate(tags):
        if tag == 'O' or tag == '<PAD>':
            if current_span:
                spans.add(tuple(current_span))
                current_span = None
        elif tag.startswith('B-'):
            if current_span:
                spans.add(tuple(current_span))
            current_span = [tag[2:], i, i]
        elif tag.startswith('I-'):
            if current_span and current_span[0] == tag[2:]:
                current_span[2] = i
            else:
                if current_span:
                    spans.add(tuple(current_span))
                current_span = [tag[2:], i, i]
                
    if current_span:
        spans.add(tuple(current_span))
    return spans

def compute_span_f1(true_tags: list[str], pred_tags: list[str]) -> float:
    """Calculates strict Span-Level F1 for ATIS."""
    true_spans = extract_spans(true_tags)
    pred_spans = extract_spans(pred_tags)
    
    correct_spans = true_spans.intersection(pred_spans)
    num_correct = len(correct_spans)
    
    precision = num_correct / len(pred_spans) if pred_spans else 0.0
    recall = num_correct / len(true_spans) if true_spans else 0.0
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


import os
import urllib.request
import zipfile


def get_glove_weights(token_to_id: dict, embedding_dim: int) -> torch.Tensor:
    """Downloads official GloVe directly from Stanford and builds the matrix."""
    glove_dir = "glove_data"
    glove_file = f"{glove_dir}/glove.6B.{embedding_dim}d.txt"
    # zip_path = f"{glove_dir}/glove.6B.zip"
    
    # # 1. Download and unzip if it doesn't exist locally
    # if not os.path.exists(glove_file):
    #     os.makedirs(glove_dir, exist_ok=True)
    #     print("Downloading official GloVe embeddings from Stanford (822MB)...")
        
    #     # Download the official Stanford zip file
    #     urllib.request.urlretrieve("https://nlp.stanford.edu/data/glove.6B.zip", zip_path)
        
    #     print("Download complete. Extracting embeddings...")
    #     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    #         zip_ref.extractall(glove_dir)
            
    #     # Clean up the zip file to save space
    #     os.remove(zip_path)
        
    print(f"Loading vectors from {glove_file}...")
    
    # 2. Parse the text file
    glove_dict = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            glove_dict[word] = np.array(values[1:], dtype='float32')
            
    # 3. Build weight matrix
    vocab_size = len(token_to_id)
    weight_matrix = torch.randn((vocab_size, embedding_dim))
    
    words_found = 0
    for word, idx in token_to_id.items():
        if word.lower() in glove_dict:
            weight_matrix[idx] = torch.tensor(glove_dict[word.lower()])
            words_found += 1
            
    print(f"Loaded {words_found}/{vocab_size} words from GloVe.")
    return weight_matrix

def train_loop(
    model: Any,
    train_data: Any,
    dev_data: Any,
    args: argparse.Namespace,
) -> float:
    """Train `model` on `train_data`, validating against `dev_data`.

    This is the part you implement.

    Your implementation must:
      - Train `model` on `train_data`.

      - Use `dev_data` for validation (and early stopping, if you choose to implement it).

      - When `args.model == "BiLSTM"`, log training progress to
        TensorBoard via `args.tensorboard_logdir` (e.g. using
        `torch.utils.tensorboard.SummaryWriter`). Not applicable for `args.model == "crf"`.

      - Save the final trained model to `args.checkpoint_dir`. See
        `evaluate.py`'s `load_model()` docstring for the exact checkpoint
        format your saved model must match.

    Returns:
        Final dev-set slot F1 (float). `main()` will print what you return.
    """
    # ------------------------------------------------------------------
    

    if args.model == "crf":
        print("Training CRF...")
        # 1. Extract features using the function you wrote earlier     
        X_train = [ex.tokens for ex in train_data]
        y_train = [ex.slots for ex in train_data]
        
        # 2. Fit the model (assuming model is an sklearn-crfsuite wrapper)
        model.fit(X_train, y_train)
        
        # 3. Evaluate on Dev
        X_dev = [ex.tokens for ex in dev_data]
        y_dev = [ex.slots for ex in dev_data]
        y_pred = model.predict(X_dev)
        
        # Flatten lists of lists into single lists for F1 evaluation
        flat_true = [tag for sentence_tags in y_dev for tag in sentence_tags]
        flat_pred = [tag for sentence_tags in y_pred for tag in sentence_tags]
        
        dev_f1 = compute_span_f1(flat_true, flat_pred)
        
        # Save model (usually via pickle or joblib for CRF)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

        save_path = os.path.join(args.checkpoint_dir, "crf_model.pkl")
        joblib.dump(model.model, save_path)
        
        return float(dev_f1)

    # ---------------------------------------------------------
    # BRANCH B: BiLSTM (PyTorch Training Loop)
    # ---------------------------------------------------------
    _, _, _, vocab = load_data(args.train_fraction, args.seed)
    token_to_id = vocab["token_to_id"]
    word_pad_idx = token_to_id.get('<PAD>', 0)
    word_unk_idx = token_to_id.get('<UNK>', 1)
    
    # 2. Build tag_to_id directly from the data_loader's official set
    # Sorting guarantees the mapping is deterministic and matches the tagset_size
    all_tags = sorted(list(data_loader.slot_label_set(train_data)))
    tag_to_id = {tag: i for i, tag in enumerate(all_tags)}
    id_to_tag = {i: tag for tag, i in tag_to_id.items()}


    def collate_fn(batch_examples):
            sentence_tensors, slot_tensors, lengths = [], [], []
            for ex in batch_examples:
                token_ids = [token_to_id.get(w, word_unk_idx) for w in ex.tokens]
                # Default to an existing tag (like 'O') if unseen, but typically dev sets don't have unknown tags
                slot_ids = [tag_to_id.get(s, 0) for s in ex.slots]
                
                sentence_tensors.append(torch.tensor(token_ids, dtype=torch.long))
                slot_tensors.append(torch.tensor(slot_ids, dtype=torch.long))
                lengths.append(len(token_ids))
                
            # Pad sentences with the official <PAD> id
            padded_sentences = torch.nn.utils.rnn.pad_sequence(sentence_tensors, batch_first=True, padding_value=word_pad_idx)
            # Pad slots with -100 so CrossEntropyLoss automatically ignores them
            padded_slots = torch.nn.utils.rnn.pad_sequence(slot_tensors, batch_first=True, padding_value=-100)
            
            lengths_tensor = torch.tensor(lengths, dtype=torch.long)
            return padded_sentences, padded_slots, lengths_tensor
    # for ex in train_data:
    #     for word in ex.tokens:
    #         if word not in word2idx: word2idx[word] = len(word2idx)
    #     for tag in ex.slots:
    #         if tag not in tag2idx: tag2idx[tag] = len(tag2idx)
            
    # idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    # pad_idx = word2idx['<PAD>']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 2. Inject GloVe Embeddings
    # Extract the model's actual embedding dimension (e.g., 100)
    embedding_dim = model.embedding.embedding_dim 
    glove_weights = get_glove_weights(token_to_id, embedding_dim)
    model.embedding.weight.data.copy_(glove_weights)
        

    # Setup DataLoaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=getattr(args, 'batch_size', 32), shuffle=True, collate_fn=collate_fn)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=getattr(args, 'batch_size', 32), shuffle=False, collate_fn=collate_fn)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=getattr(args, 'learning_rate', 0.001))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    writer = None
    if getattr(args, "tensorboard_logdir", None):
        os.makedirs(args.tensorboard_logdir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.tensorboard_logdir)
        
    best_dev_f1 = 0.0
    
    for epoch in range(getattr(args, 'epochs', 5)):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        
        for sentences, slots, lengths in train_loader:
            sentences, slots = sentences.to(device), slots.to(device)
            
            optimizer.zero_grad()
            
            # Pass both the data and the lengths to the model
            logits = model(sentences, lengths)
            loss = criterion(logits.view(-1, logits.shape[-1]), slots.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            
        # --- EVALUATION PHASE ---
        model.eval()
        val_loss = 0.0
        all_true_tags = []
        all_pred_tags = []
        
        with torch.no_grad():
            for sentences, slots, lengths in dev_loader:
                sentences, slots = sentences.to(device), slots.to(device)
                logits = model(sentences, lengths)
                loss = criterion(logits.view(-1, logits.shape[-1]), slots.view(-1))
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=-1)
                preds_list, slots_list = preds.cpu().tolist(), slots.cpu().tolist()
                
                for i in range(len(slots_list)):
                    for j in range(len(slots_list[i])):
                        if slots_list[i][j] != -100:
                            all_true_tags.append(id_to_tag[slots_list[i][j]])
                            all_pred_tags.append(id_to_tag[preds_list[i][j]])
                            
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(dev_loader)
        current_f1 = compute_span_f1(all_true_tags, all_pred_tags)

        checkpoint = {
            "state_dict": model.state_dict(),
            "vocab_size": len(token_to_id),  # or however your vocab size is stored
            "tagset_size": len(id_to_tag),   # or however your tagset size is stored
            "vocab": {
                "token_to_id": token_to_id,
                "id_to_tag": id_to_tag
            }
        }

       
        if writer:
            # writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            # writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
            writer.add_scalar('Metric/F1', current_f1, epoch)
            
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Dev F1: {current_f1:.4f}")
        
        # --- CHECKPOINTING (Early Stopping based on F1) ---
        if current_f1 > best_dev_f1:
            best_dev_f1 = current_f1
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            # Adjust the saved dict based on evaluate.py's load_model() expectations
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, "bilstm_model.pt"))
         
            
    if writer:
        writer.close()
        
    return float(best_dev_f1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_data, dev_data, test_data, vocab = load_data(args.train_fraction, args.seed)

    if args.model == "bilstm":
        model = build_model(
            args.model, vocab_size=vocab["vocab_size"], tagset_size=vocab["tagset_size"]
        )

    else:
        model = build_model(args.model)

    dev_score = train_loop(model, train_data, dev_data, args)

    import os

    if not os.path.isdir(args.checkpoint_dir) or not os.listdir(args.checkpoint_dir):
        raise RuntimeError(
            f"Expected train_loop() to save a checkpoint to {args.checkpoint_dir!r}, "
            "but the directory is missing or empty."
        )

    print("Training complete.")
    print(f"  Model:          {args.model}")
    print(f"  Train fraction: {args.train_fraction}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")

    if dev_score is not None:
        print(f"  Final dev F1:   {dev_score:.4f}")


if __name__ == "__main__":
    main()
