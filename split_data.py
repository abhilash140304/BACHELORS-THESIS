import json
import random
import os

# =========================
# CONFIG
# =========================
INPUT_FILE = "processed_sequences.json"
OUTPUT_DIR = "split_data"

SPLIT_RATIO = 0.6   # 👈 change to 0.6 / 0.5 for better evaluation
SEED = 42

USE_K_FOLD = True
K_FOLDS = 5

# =========================
# LOAD DATA
# =========================
with open(INPUT_FILE, "r") as f:
    data = json.load(f)

session_ids = data["session_ids"]
sequences = data["sequences"]
encoded_sequences = data["encoded_sequences"]
time_gaps = data["time_gaps_seconds"]

n_sessions = len(session_ids)

# =========================
# HELPER FUNCTION
# =========================
def subset(idxs):
    return {
        "session_ids": [session_ids[i] for i in idxs],
        "sequences": [sequences[i] for i in idxs],
        "encoded_sequences": [encoded_sequences[i] for i in idxs],
        "time_gaps_seconds": [time_gaps[i] for i in idxs]
    }

# =========================
# SHUFFLE
# =========================
random.seed(SEED)
indices = list(range(n_sessions))
random.shuffle(indices)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# OPTION 1: STANDARD TRAIN/TEST SPLIT
# =========================================================
if not USE_K_FOLD:
    split_index = int(SPLIT_RATIO * n_sessions)

    train_idx = indices[:split_index]
    test_idx = indices[split_index:]

    train_data = subset(train_idx)
    test_data = subset(test_idx)

    with open(os.path.join(OUTPUT_DIR, "train_sequences.json"), "w") as f:
        json.dump(train_data, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "test_sequences.json"), "w") as f:
        json.dump(test_data, f, indent=2)

    print("✅ Standard split done!")
    print(f"Train sessions: {len(train_idx)}")
    print(f"Test sessions : {len(test_idx)}")

# =========================================================
# OPTION 2: K-FOLD CROSS VALIDATION (RECOMMENDED)
# =========================================================
else:
    fold_size = n_sessions // K_FOLDS

    print(f"✅ Creating {K_FOLDS}-fold cross-validation splits...\n")

    for fold in range(K_FOLDS):
        start = fold * fold_size
        end = start + fold_size if fold != K_FOLDS - 1 else n_sessions

        test_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]

        fold_dir = os.path.join(OUTPUT_DIR, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)

        train_data = subset(train_idx)
        test_data = subset(test_idx)

        with open(os.path.join(fold_dir, "train_sequences.json"), "w") as f:
            json.dump(train_data, f, indent=2)

        with open(os.path.join(fold_dir, "test_sequences.json"), "w") as f:
            json.dump(test_data, f, indent=2)

        print(f"Fold {fold+1}:")
        print(f"  Train sessions: {len(train_idx)}")
        print(f"  Test sessions : {len(test_idx)}\n")

    print("🎯 K-Fold splits created successfully!")