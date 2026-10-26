"""Train and save the local demonstration model artifacts."""

from pathlib import Path

from modeling import save_model_bundle, train_models


def main() -> None:
    root = Path(__file__).resolve().parent
    dataset = root / "indian_agriculture_data.csv"
    bundle, metrics = train_models(dataset)
    artifact = save_model_bundle(bundle)
    print(f"Saved model artifact to {artifact.relative_to(root)}")
    print(f"Dataset SHA-256: {bundle['dataset_sha256']}")
    print(f"scikit-learn: {bundle['sklearn_version']}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("Metrics are for a single synthetic-data holdout and are not evidence of real-world performance.")


if __name__ == "__main__":
    main()
