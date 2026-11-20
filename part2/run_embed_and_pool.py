from datetime import datetime
import json
import os
from pathlib import Path

import numpy as np
import torch

from src import CLASSES, download_pretrained, load_sst2
from part1.experiment import cross_validation_split
from part2.clslib import BasicTokenizer, PretrainedEmbeddings
from part2.experiment import cross_validation


def main():
    os.chdir(Path(__file__).resolve().parent)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir = Path("results/embed_and_pool/cross_validation/frozen_glove_embed") / run_id
    output_dir.mkdir(parents=True)
    print(f"Output: {output_dir.resolve()}", flush=True)

    archive, dimensions, skip_first_line = download_pretrained("glove")
    embeddings = PretrainedEmbeddings(archive, dimensions, skip_first_line=skip_first_line)
    tokenizer = BasicTokenizer(
        lowercase=True,
        remove_punct=False,
        remove_stopwords=False,
        vocab=embeddings.vocab,
    )
    train, _, _ = load_sst2()
    splits = cross_validation_split(pairs=train, k=5, seed=0)
    results = cross_validation(
        outcome_space=CLASSES,
        data_splits=splits,
        tokenizer=tokenizer,
        emb_model=embeddings,
        enc_type="embed-and-pool",
        enc_kwargs={"freeze_embeddings": False},
        regressor_kwargs={"hidden_size1": 256, "hidden_size2": 128, "dropout": 0.3},
        opt_kwargs={
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "batch_size": 64,
            "num_epochs": 10,
        },
        device=device,
        save_to=str(output_dir),
    )

    accuracies = [float(value) for value in results["accuracies"].values()]
    metrics = {
        "accuracy_per_fold": accuracies,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
    }
    metrics_path = output_dir / "accuracy.json"
    metrics_path.write_text(json.dumps(metrics, indent=4) + "\n", encoding="utf-8")
    print(f"Accuracy per fold: {accuracies}")
    print(f"Mean accuracy: {metrics['mean_accuracy']:.4f}")
    print(f"Standard deviation: {metrics['std_accuracy']:.4f}")
    print(f"Metrics: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
