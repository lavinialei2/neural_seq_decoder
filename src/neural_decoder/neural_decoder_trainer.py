import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .model import GRUDecoder
from .dataset import SpeechDataset


def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

def _apply_time_mask(batch, mask_count, max_length):
    if mask_count <= 0 or max_length <= 0:
        return batch

    batch_size, time_steps, _ = batch.shape
    if time_steps == 0:
        return batch

    for b in range(batch_size):
        for _ in range(mask_count):
            mask_len = int(
                torch.randint(1, max_length + 1, (1,), device=batch.device).item()
            )
            mask_len = min(mask_len, time_steps)
            if mask_len <= 0:
                continue
            max_start = max(1, time_steps - mask_len + 1)
            start = int(
                torch.randint(0, max_start, (1,), device=batch.device).item()
            )
            end = min(time_steps, start + mask_len)
            batch[b, start:end, :] = 0
    return batch

def _apply_feature_mask(batch, mask_count, mask_size):
    if mask_count <= 0 or mask_size <= 0:
        return batch
    batch_size, time_steps, feat_dim = batch.shape
    if feat_dim == 0:
        return batch
    for b in range(batch_size):
        for _ in range(mask_count):
            size = min(int(mask_size), feat_dim)
            if size <= 0:
                continue
            start = int(torch.randint(0, max(1, feat_dim - size + 1), (1,), device=batch.device).item())
            batch[b, :, start : start + size] = 0
    return batch

def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_smoothing = args.get("label_smoothing", 0.0)
    grad_clip = args.get("grad_clip", 0.0)
    warmup_steps = args.get("warmup_steps", 0)
    eval_every = args.get("eval_every", 100)
    time_mask_count = args.get("time_mask_count", 0)
    time_mask_max_length = args.get("time_mask_max_length", 0)
    patience = args.get("early_stopping_patience", None)
    early_stopping_start = args.get("early_stopping_start", args["nBatch"])
    optimizer_name = args.get("optimizer", "adam").lower()
    adam_epsilon = args.get("adam_epsilon", 1e-8)
    feature_mask_count = args.get("feature_mask_count", 0)
    feature_mask_size = args.get("feature_mask_size", 0)

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
        rnn_type=args.get("rnn_type", "gru"),
        post_ffn_layers=args.get("post_ffn_layers", 0),
        post_ffn_hidden=args.get("post_ffn_hidden", None),
        post_ffn_dropout=args.get("post_ffn_dropout", 0.0),
    ).to(device)

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args["lrStart"],
            betas=(0.9, 0.999),
            eps=adam_epsilon,
            weight_decay=args["l2_decay"],
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args["lrStart"],
            betas=(0.9, 0.999),
            eps=adam_epsilon,
            weight_decay=args["l2_decay"],
        )
    end_factor = args["lrEnd"] / args["lrStart"]

    def lr_schedule(step_idx):
        if warmup_steps > 0 and step_idx < warmup_steps:
            return max(1e-8, float(step_idx + 1) / float(warmup_steps))
        decay_steps = max(1, args["nBatch"] - max(warmup_steps, 1))
        progress = max(0.0, float(step_idx - warmup_steps)) / decay_steps
        return max(end_factor, 1.0 - progress * (1.0 - end_factor))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    # --train--
    testLoss = []
    testCER = []
    startTime = time.time()
    writer = SummaryWriter(log_dir=os.path.join(args["outputDir"], "tensorboard"))
    global_step = 0
    best_cer = float("inf")
    batches_since_improvement = 0
    train_iter = iter(trainLoader)

    try:
        for batch in range(args["nBatch"]):
            model.train()
            try:
                X, y, X_len, y_len, dayIdx = next(train_iter)
            except StopIteration:
                train_iter = iter(trainLoader)
                X, y, X_len, y_len, dayIdx = next(train_iter)
            X, y, X_len, y_len, dayIdx = (
                X.to(device),
                y.to(device),
                X_len.to(device),
                y_len.to(device),
                dayIdx.to(device),
            )

            # Noise augmentation is faster on GPU
            if args["whiteNoiseSD"] > 0:
                X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

            if args["constantOffsetSD"] > 0:
                X += (
                    torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                    * args["constantOffsetSD"]
                )

            if feature_mask_count > 0 and feature_mask_size > 0:
                X = _apply_feature_mask(X, feature_mask_count, feature_mask_size)
            if time_mask_count > 0 and time_mask_max_length > 0:
                X = _apply_time_mask(X, time_mask_count, time_mask_max_length)

            # Compute prediction error
            pred = model.forward(X, dayIdx)
            log_probs = pred.log_softmax(2)
            input_lengths = torch.clamp(
                torch.floor((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                min=1,
            )

            ctc_loss = loss_ctc(
                torch.permute(log_probs, [1, 0, 2]),
                y,
                input_lengths,
                y_len,
            )
            if label_smoothing > 0:
                smoothing_loss = -log_probs.mean()
                loss = (1 - label_smoothing) * ctc_loss + label_smoothing * smoothing_loss
            else:
                loss = ctc_loss

            if not torch.isfinite(loss):
                print(f"Skipping batch {batch} due to non-finite loss")
                optimizer.zero_grad()
                continue

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            if grad_clip and grad_clip > 0:
                clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1
            writer.add_scalar("train/ctc_loss", ctc_loss.item(), global_step)
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            # Eval
            if batch % eval_every == 0:
                with torch.no_grad():
                    model.eval()
                    allLoss = []
                    total_edit_distance = 0
                    total_seq_length = 0
                    for X, y, X_len, y_len, testDayIdx in testLoader:
                        X, y, X_len, y_len, testDayIdx = (
                            X.to(device),
                            y.to(device),
                            X_len.to(device),
                            y_len.to(device),
                            testDayIdx.to(device),
                        )

                        pred = model.forward(X, testDayIdx)
                        eval_lengths = torch.clamp(
                            torch.floor((X_len - model.kernelLen) / model.strideLen).to(
                                torch.int32
                            ),
                            min=1,
                        )
                        eval_loss = loss_ctc(
                            torch.permute(pred.log_softmax(2), [1, 0, 2]),
                            y,
                            eval_lengths,
                            y_len,
                        )
                        allLoss.append(eval_loss.cpu().detach().numpy())

                        for iterIdx in range(pred.shape[0]):
                            decodedSeq = torch.argmax(
                                torch.tensor(
                                    pred[iterIdx, 0 : eval_lengths[iterIdx], :]
                                ),
                                dim=-1,
                            )  # [num_seq,]
                            decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                            decodedSeq = decodedSeq.cpu().detach().numpy()
                            decodedSeq = np.array([i for i in decodedSeq if i != 0])

                            trueSeq = np.array(
                                y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                            )

                            matcher = SequenceMatcher(
                                a=trueSeq.tolist(), b=decodedSeq.tolist()
                            )
                            total_edit_distance += matcher.distance()
                            total_seq_length += len(trueSeq)

                    avgDayLoss = np.sum(allLoss) / len(testLoader)
                    cer = total_edit_distance / total_seq_length

                    endTime = time.time()
                    print(
                        f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/max(1, eval_every):>7.3f}"
                    )
                    writer.add_scalar("eval/ctc_loss", avgDayLoss, global_step)
                    writer.add_scalar("eval/cer", cer, global_step)
                    startTime = time.time()

                if cer < best_cer:
                    torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
                    best_cer = cer
                    batches_since_improvement = 0
                else:
                    batches_since_improvement += 1

                testLoss.append(avgDayLoss)
                testCER.append(cer)

                tStats = {}
                tStats["testLoss"] = np.array(testLoss)
                tStats["testCER"] = np.array(testCER)

                with open(args["outputDir"] + "/trainingStats", "wb") as file:
                    pickle.dump(tStats, file)

                if (
                    patience is not None
                    and batch >= early_stopping_start
                    and batches_since_improvement >= patience
                ):
                    print(
                        f"Early stopping triggered at batch {batch} with best CER {best_cer:.4f}"
                    )
                    break
    finally:
        writer.close()


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
        rnn_type=args.get("rnn_type", "gru"),
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)

if __name__ == "__main__":
    main()
