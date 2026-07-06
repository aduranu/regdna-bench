"""probing task: train a lightweight probe on frozen last-layer embeddings.

measures how decodable a benchmark model's representations are for a downstream
activity target (e.g. lentiMPRA expression). model-agnostic: it only uses the
BenchModel embedding-extraction hook, so any wrapper implementing
add_last_layer_embedding_extraction + exposing .last_layer_embedding can be
probed (see examples/d3/wrapper.py).

two probe heads are supported, both ported from the original D3 probing scripts
(Score-Entropy-Discrete-Diffusion/PL_mpra_{ridge,cnn}.py):
  - ridge: mean-pool the (L, hidden) embedding -> (hidden,), sklearn RidgeCV
  - cnn:   keep the full (L, hidden) embedding, RepCNN conv stack regressor
the only thing that changes across a model comparison is the wrapped model; data,
token convention and splits stay identical.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# probe inputs/targets are fit in fp32; embeddings are pooled/stacked in fp32 too
EMBED_DTYPE = torch.float32

# the 768-d embeddings are strongly collinear, so a too-small alpha leaves the
# ridge solve ill-conditioned; RidgeCV picks a properly-sized alpha by LOO cv
DEFAULT_ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)

# cnn-probe training hyperparameters, matching PL_mpra_cnn.py exactly
CNN_LR = 1e-4
CNN_BATCH_SIZE = 256
CNN_MAX_EPOCHS = 100
CNN_PATIENCE = 10
CNN_SEED = 0


def extract_embeddings(model, sequences, batch_size=256, pool=True):
    # run the model in batches and collect last-layer embeddings. the model must
    # have its extraction hook installed and expose .last_layer_embedding as
    # (B, L, hidden). pool=True mean-pools to (B, hidden) for the ridge probe;
    # pool=False keeps the full (B, L, hidden) tensor for the cnn probe.
    #
    # the unpooled cnn cache is large (e.g. 314k x 230 x 768), so it is stored in
    # fp16 and written into a preallocated buffer to keep peak host memory at one
    # copy (a list+cat would transiently double it). per-batch fp32 cast happens
    # at train time. pooled embeddings are tiny and stay fp32.
    model.add_last_layer_embedding_extraction()

    store_dtype = EMBED_DTYPE if pool else torch.float16

    n = len(sequences)
    buffer = None
    for start in range(0, n, batch_size):
        batch = sequences[start:start + batch_size]

        model.forward(batch)
        embedding = model.last_layer_embedding

        out = embedding.mean(dim=1) if pool else embedding
        out = out.to(store_dtype).cpu()

        if buffer is None:
            buffer = torch.empty((n, *out.shape[1:]), dtype=store_dtype)

        buffer[start:start + out.shape[0]] = out

    return buffer


def fit_linear_probe(x_train, y_train, x_test, alphas=DEFAULT_ALPHAS):
    # standardize then ridge-regress; deliberately simple so the score reflects
    # the representation rather than probe capacity. RidgeCV tunes alpha by
    # leave-one-out cv, which also conditions the otherwise near-singular solve.
    scaler = StandardScaler().fit(x_train)

    probe = RidgeCV(alphas=alphas).fit(scaler.transform(x_train), y_train)
    y_pred = probe.predict(scaler.transform(x_test))

    return y_pred, float(probe.alpha_)


def run_probing(model, train_sequences, train_labels, test_sequences, test_labels, alphas=DEFAULT_ALPHAS):
    x_train = extract_embeddings(model, train_sequences).numpy()
    x_test = extract_embeddings(model, test_sequences).numpy()

    y_pred, best_alpha = fit_linear_probe(x_train, np.asarray(train_labels), x_test, alphas=alphas)

    y_true = np.asarray(test_labels)
    pearson = float(pearsonr(y_pred.ravel(), y_true.ravel())[0])
    spearman = float(spearmanr(y_pred.ravel(), y_true.ravel())[0])

    print(f"ridge probing complete: pearson={pearson:.4f} spearman={spearman:.4f} alpha={best_alpha:g}")

    return {"pearson": pearson, "spearman": spearman, "alpha": best_alpha}


class RepCNN(nn.Module):
    # cnn probe over the full (L, hidden) representation, ported verbatim from
    # PL_mpra_cnn.py (a torch re-impl of the keras ResidualBind-style head): a
    # 1x1 channel reduction then two conv blocks (exp then relu activation) and a
    # two-layer mlp readout. input_shape is (L, hidden).
    def __init__(self, input_shape, output_shape=1, factor=1):
        super().__init__()

        self.config = {
            "reduce_dim": 196, "conv1_filter": 196, "conv1_kernel": 7,
            "activation": "exponential", "dropout1": 0.2, "res_pool": 5,
            "res_dropout": 0.2, "conv2_filter": 256, "conv2_kernel": 7,
            "pool2_size": 4, "dropout2": 0.2, "dense": 512, "dense2": 256,
            "l_rate": 1e-4,
        }
        self.factor = factor

        length, hidden = input_shape[0], input_shape[1]

        self.batch_norm1 = nn.BatchNorm1d(hidden)
        self.conv1d_reduce = nn.Conv1d(hidden, self.config["reduce_dim"], kernel_size=1)

        self.conv1 = nn.Conv1d(self.config["reduce_dim"], self.config["conv1_filter"] * factor,
                               kernel_size=self.config["conv1_kernel"], padding="same")
        self.batch_norm2 = nn.BatchNorm1d(self.config["conv1_filter"] * factor)
        self.dropout1 = nn.Dropout(self.config["dropout1"])
        self.max_pool1 = nn.MaxPool1d(kernel_size=self.config["res_pool"])

        self.conv2 = nn.Conv1d(self.config["conv1_filter"] * factor, self.config["conv2_filter"] * factor,
                               kernel_size=self.config["conv2_kernel"], padding="same")
        self.batch_norm3 = nn.BatchNorm1d(self.config["conv2_filter"] * factor)
        self.activation2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool1d(kernel_size=self.config["pool2_size"])
        self.dropout2 = nn.Dropout(self.config["dropout2"])

        flattened_size = (length // self.config["res_pool"] // self.config["pool2_size"]) * self.config["conv2_filter"] * factor

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(flattened_size, self.config["dense"] * factor)
        self.batch_norm4 = nn.BatchNorm1d(self.config["dense"] * factor)
        self.activation3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)

        self.dense2 = nn.Linear(self.config["dense"] * factor, self.config["dense2"] * factor)
        self.batch_norm5 = nn.BatchNorm1d(self.config["dense2"] * factor)
        self.activation4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.5)

        self.output_layer = nn.Linear(self.config["dense2"] * factor, output_shape)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.005)

    def forward(self, x):
        # (B, L, hidden) -> conv1d wants (B, hidden, L)
        x = x.transpose(1, 2)
        x = self.batch_norm1(x)
        x = self.conv1d_reduce(x)

        x = self.conv1(x)
        x = self.batch_norm2(x)
        x = torch.exp(x)
        x = self.dropout1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.batch_norm3(x)
        x = self.activation2(x)
        x = self.max_pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batch_norm4(x)
        x = self.activation3(x)
        x = self.dropout3(x)

        x = self.dense2(x)
        x = self.batch_norm5(x)
        x = self.activation4(x)
        x = self.dropout4(x)

        return self.output_layer(x)


def _cnn_epoch(probe, loader, criterion, optimizer, device):
    # one train pass when optimizer is given, else one eval pass; returns the
    # sample-weighted mean loss over the loader
    train = optimizer is not None
    probe.train() if train else probe.eval()

    total = 0.0
    with torch.set_grad_enabled(train):
        for inputs, targets in loader:
            # embeddings are cached fp16 to save host memory; cast to fp32 here
            inputs = inputs.to(device).to(EMBED_DTYPE)
            targets = targets.to(device)

            if train:
                optimizer.zero_grad()

            loss = criterion(probe(inputs), targets)

            if train:
                loss.backward()
                optimizer.step()

            total += loss.item() * inputs.size(0)

    return total / len(loader.dataset)


def fit_cnn_probe(x_train, y_train, x_valid, y_valid, x_test, device="cuda"):
    # train the RepCNN head on full-length embeddings with early stopping on the
    # validation split, then predict the test split. mirrors train_repcnn in
    # PL_mpra_cnn.py (mse, adam 1e-4, ReduceLROnPlateau, patience 10).
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(CNN_SEED)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=CNN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(TensorDataset(x_valid, y_valid), batch_size=CNN_BATCH_SIZE)

    probe = RepCNN(x_train.shape[1:], output_shape=y_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=CNN_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, min_lr=1e-8)

    best_loss, best_state, patience_counter = float("inf"), None, 0
    for _epoch in range(CNN_MAX_EPOCHS):
        _cnn_epoch(probe, train_loader, criterion, optimizer, device)
        valid_loss = _cnn_epoch(probe, valid_loader, criterion, None, device)
        scheduler.step(valid_loss)

        if valid_loss < best_loss:
            best_loss, best_state, patience_counter = valid_loss, {k: v.detach().clone() for k, v in probe.state_dict().items()}, 0
        else:
            patience_counter += 1
            if patience_counter >= CNN_PATIENCE:
                break

    if best_state is not None:
        probe.load_state_dict(best_state)

    probe.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, x_test.shape[0], CNN_BATCH_SIZE):
            chunk = x_test[start:start + CNN_BATCH_SIZE].to(device).to(EMBED_DTYPE)
            preds.append(probe(chunk).cpu())

    y_pred = torch.cat(preds, dim=0).numpy()

    return y_pred, best_loss


def run_cnn_probing(model, train_sequences, train_labels, valid_sequences, valid_labels,
                    test_sequences, test_labels, device="cuda"):
    x_train = extract_embeddings(model, train_sequences, pool=False)
    x_valid = extract_embeddings(model, valid_sequences, pool=False)
    x_test = extract_embeddings(model, test_sequences, pool=False)

    y_train = torch.as_tensor(np.asarray(train_labels), dtype=EMBED_DTYPE)
    y_valid = torch.as_tensor(np.asarray(valid_labels), dtype=EMBED_DTYPE)

    y_pred, best_loss = fit_cnn_probe(x_train, y_train, x_valid, y_valid, x_test, device=device)

    y_true = np.asarray(test_labels)
    pearson = float(pearsonr(y_pred.ravel(), y_true.ravel())[0])
    spearman = float(spearmanr(y_pred.ravel(), y_true.ravel())[0])

    print(f"cnn probing complete: pearson={pearson:.4f} spearman={spearman:.4f} val_mse={best_loss:.4f}")

    return {"pearson": pearson, "spearman": spearman, "val_mse": best_loss}
