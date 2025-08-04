#!/usr/bin/env python
# coding: utf-8

### Experiment 019-4
# - Model: Qwen/Qwen3-Embedding-8B

import os
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, make_scorer, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import time
import pickle
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from transformers.utils import is_flash_attn_2_available
import wandb
from wandb import AlertLevel



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["WANDB_PROJECT"] = "GermEval2025-Substask1"
os.environ["WANDB_LOG_MODEL"] = "false"

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print("CUDA not available, using CPU")

experiment_name = "exp019-4"

testing_mode = False


# Load data
comments = pd.read_csv("../../share-GermEval2025-data/Data/training data/comments.csv")
task1 = pd.read_csv("../../share-GermEval2025-data/Data/training data/task1.csv")
comments = comments.merge(task1, on=["document", "comment_id"])

# Remove duplicates
df = comments.drop_duplicates(subset=['comment', 'flausch'])
df.reset_index(drop=True, inplace=True)

# Use only a small subset for testing
if testing_mode:
    os.environ["WANDB_MODE"] = "offline"
    testing_mode_sample_size = 1000
    df = df.sample(n=testing_mode_sample_size, random_state=42).reset_index(drop=True)
    print(f"Testing mode: using only {testing_mode_sample_size} samples for quick testing.")

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

class Qwen3Embedder:
    def __init__(self, model_name='Qwen/Qwen3-Embedding-8B', instruction=None, max_length=1024):
        if instruction is None:
            instruction = 'Classify a given comment as either flausch (a positive, supportive expression) or non-flausch.'
        self.instruction = instruction

        if is_flash_attn_2_available():
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        else:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)

        self.model = self.model.cuda()
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
        self.max_length = max_length

    def get_detailed_instruct(self, query: str) -> str:
        return f'Instruct: {self.instruction}\nQuery:{query}'

    def encode_batch(self, texts, batch_size=32):
        """Encode texts in batches to handle memory efficiently"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = [self.get_detailed_instruct(comment) for comment in texts[i:i + batch_size]]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(device)

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
                #embeddings = embeddings.float()

            all_embeddings.append(embeddings.cpu().numpy())

        # Normalize embeddings (sollte ich?)
        #import torch.nn.functional as F
        #output = F.normalize(all_embeddings, p=2, dim=1)
        return np.vstack(all_embeddings)

# Initialize embedder
print("Loading Qwen3 Embeddings v3...")
embedder = Qwen3Embedder(instruction='Classify a given comment as either flausch (a positive, supportive expression) or non-flausch')

X, y = df["comment"], df["flausch"].map(dict(yes=1, no=0))

# load embeddings if they exist
embeddings_file = f'{"testing_" if testing_mode else ""}Qwen3-Embedding-8B-{experiment_name}.npy'
if os.path.exists(embeddings_file):
    print(f"Loading existing embeddings from {embeddings_file}")
    X_embeddings = np.load(embeddings_file)
else:
    print("Embeddings not found, generating new embeddings...")
    # Encode texts in batches to avoid memory issues
    X_embeddings = embedder.encode_batch(X.tolist(), batch_size=64)
    print(f"Generated embeddings with shape: {X_embeddings.shape}")

    # save embeddings to avoid recomputation
    np.save(embeddings_file, X_embeddings)

wandb.init(
    project=os.environ["WANDB_PROJECT"],
    dir='./wandb_logs',
    name=f"{experiment_name}",
)

# 5-fold stratified cross-validation
kf_splits = 5

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(random_state=42, cache_size=2000))
])

param_grid = [
    {
        # Fitting 5 folds for each of 25 candidates, totalling 125 fits
        'svm__kernel': ['rbf'],
        'svm__C': [5, 6, 7, 8, 9, 10],
        'svm__gamma': [0.00008, 0.0001, 0.0002, 1/4096, 0.0003, 0.0004, 0.0005, 0.0006]
        # wähle diesen Bereich, da wir mit Qwen3-Embedding-8B 4096 Dimensionen haben
        # und wir bei auto bei 1/4096 also ca. 2.4e-4 landen würden
    },
#    {
#        'kernel': ['poly'],
#        'C': [0.1, 1, 10, 100],
#        'degree': [2, 3, 4],
#        'gamma': ['scale', 'auto', 0.001, 0.01],
#        'coef0': [0.0, 0.1, 0.5, 1]
#    }
]


f1_pos_scorer = make_scorer(f1_score, pos_label=1, average='binary')

X_train = X_embeddings
y_train = y

# 5‐fach StratifiedCV für die Grid‐Search
cv_inner = StratifiedKFold(n_splits=kf_splits, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv_inner,
    scoring=f1_pos_scorer,
    n_jobs=63,
    verbose=3,
    return_train_score=True
)

grid.fit(X_train, y_train)

# 6. Ergebnisse ausgeben
print("Best F1 (pos) auf CV:", grid.best_score_)
print("Beste Parameter:", grid.best_params_)
print("Best estimator:", grid.best_estimator_)


with open(f'scores.{experiment_name}.txt', 'a') as f:
    f.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {kf_splits}Fold CV\n')
    f.write(f'[{experiment_name}] Best F1 (pos) auf CV: {grid.best_score_}\n')
    f.write(f'[{experiment_name}] Beste Parameter: {grid.best_params_}\n')
    f.write(f'[{experiment_name}] Best estimator: {grid.best_estimator_}\n')

results = pd.DataFrame(grid.cv_results_).sort_values("rank_test_score")
print("grid.cv_results_:")
print(results)
results.to_csv(f'grid_cv_results.{experiment_name}.csv', index=False)

with open(f"grid_cv.{experiment_name}.pkl", "wb") as f:
    pickle.dump(grid, f)

print(f"GridSearchCV results saved to grid_cv_results.{experiment_name}.csv")

print(f"Training completed with {len(X_train)} samples...")


print("Experiment completed!")

wandb.alert(
    title=f'Experiment {experiment_name} finished!',
    text=f'Best F1 (pos): {grid.best_score_:.4f}\nBest Params: {grid.best_params_}',
    level=AlertLevel.INFO
)
wandb.finish()
print("Notification sent via Weights & Biases.")