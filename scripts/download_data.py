"""Download any Hugging Face text-classification dataset into the CSV layout this repo reads.

No dataset id is hardcoded on purpose: this repo does not vouch for a specific dataset.
Pick one yourself, check its license and language, then run for example:

    python scripts/download_data.py --hf-id <owner/name> --text-col <text column> \
        --label-col <label column> --out data/raw/dataset.csv

Output columns: text, label (class names; ClassLabel integers are mapped back to names).
Add a `script` column yourself if you know the script of each row; otherwise it is detected
at load time. The dataset's license is printed when the hub metadata has one, but always
read the dataset card: that metadata can be missing or wrong. Needs network access to
huggingface.co (Colab, Kaggle or your own machine) and `pip install datasets`.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hf-id", required=True, help="dataset id on the Hugging Face hub")
    parser.add_argument("--config-name", default=None, help="dataset configuration, if any")
    parser.add_argument("--splits", nargs="*", default=None, help="splits to merge (default: all)")
    parser.add_argument("--text-col", required=True)
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--out", default="data/raw/dataset.csv")
    parser.add_argument("--max-rows", type=int, default=0, help="optional cap, 0 keeps all")
    args = parser.parse_args()

    try:
        from datasets import ClassLabel, load_dataset
    except ImportError as err:
        raise SystemExit("Install the datasets package first: pip install datasets") from err

    dataset = load_dataset(args.hf_id, args.config_name)
    split_names = args.splits or list(dataset.keys())
    unknown = [s for s in split_names if s not in dataset]
    if unknown:
        raise SystemExit(f"Unknown split(s) {unknown}; available: {list(dataset.keys())}")

    frames = []
    for name in split_names:
        split = dataset[name]
        for col in (args.text_col, args.label_col):
            if col not in split.column_names:
                raise SystemExit(f"Column {col!r} not in {args.hf_id}; found {split.column_names}")
        frame = split.to_pandas()[[args.text_col, args.label_col]]
        frame.columns = ["text", "label"]
        feature = split.features[args.label_col]
        if isinstance(feature, ClassLabel):
            frame["label"] = frame["label"].map(lambda i, names=feature.names: names[i])
        frames.append(frame)

    import pandas as pd

    table = pd.concat(frames, ignore_index=True)
    if args.max_rows:
        table = table.sample(n=min(args.max_rows, len(table)), random_state=0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    info = next(iter(dataset.values())).info
    print(f"Wrote {len(table)} rows to {out}")
    print(f"Label counts:\n{table['label'].value_counts().to_string()}")
    print(
        f"License metadata on the hub: {info.license or 'not stated'}. Verify on the dataset card."
    )


if __name__ == "__main__":
    main()
