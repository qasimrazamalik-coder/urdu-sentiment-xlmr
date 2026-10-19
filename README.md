# Urdu sentiment: XLM-RoBERTa versus a TF-IDF baseline

Fine-tunes a multilingual transformer (XLM-RoBERTa base, or multilingual BERT) for Urdu sentiment, compares it with a strong word + character n-gram linear baseline on one fixed split, and reports errors separately for Urdu script and Roman Urdu (Urdu typed in Latin letters).

## Results

| Model | Accuracy | Macro-F1 | Macro-F1 95% CI | F1 Urdu script | F1 Roman Urdu |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + linear baseline | not yet run | not yet run | not yet run | not yet run | not yet run |
| xlm-roberta-base | not yet run | not yet run | not yet run | not yet run | not yet run |

Transformer minus baseline (macro-F1, paired bootstrap 95% CI): not yet run.

Fill with `make train && make eval`. The table is written to `results/results.md` and the full numbers to `results/metrics.json` by `urdu_sentiment.evaluate`; nothing here is typed by hand. `make smoke` also writes these files, but from synthetic template sentences and a tiny random model, and marks them as synthetic. Those numbers are not results.

## Demo

Screenshot added after first run: `docs/demo.png`.

## What question this answers

Does fine-tuning a multilingual transformer beat a well-tuned classical baseline on Urdu sentiment, and is any gain the same for Urdu script and for Roman Urdu? Roman Urdu has no standard spelling, and multilingual models see far less of it in pretraining than Arabic-script Urdu, so the two scripts are a natural place for the models to differ. The comparison reports uncertainty (bootstrap intervals) because test sets in this area are small.

## Quickstart

```bash
git clone https://github.com/qasimrazamalik-coder/urdu-sentiment-xlmr.git
cd urdu-sentiment-xlmr
python -m venv .venv && source .venv/bin/activate
make install
make test     # unit tests on CPU, about 15 seconds
make smoke    # end-to-end run on SYNTHETIC data with a tiny random model (not a result)

# real run: choose a dataset first, see "Choosing a dataset"
make data HF_ID=<owner/name> TEXT_COL=<text column> LABEL_COL=<label column>
make train    # baseline + xlm-roberta-base; writes predictions and checkpoints
make eval     # metrics.json, results.md, error_analysis.json, errors.csv
make app      # Gradio demo at http://127.0.0.1:7860
```

For multilingual BERT instead: `python -m urdu_sentiment.train --config configs/mbert.yaml`, then the same two commands for `evaluate` and `error_analysis` with that config. Outputs go to `results/mbert/`.

### Choosing a dataset

This repo deliberately does not hardcode a dataset id. Pick one and check it yourself:

1. Search the Hugging Face hub for Urdu sentiment or text-classification datasets, and also look for Roman Urdu ones. You need both scripts for the per-script comparison to mean anything.
2. Open the dataset card. Confirm: the language and script, how labels were produced (human annotators or heuristics such as star ratings), the label set, the license, and whether redistribution or research use is allowed.
3. Inspect a few dozen rows after download. Check that the text column holds the text and that labels look right.
4. Run `make data` with `HF_ID`, `TEXT_COL` and `LABEL_COL`. The script merges all splits (the repo makes its own split), maps class ids back to names, prints label counts and the license metadata from the hub. That metadata can be wrong or missing, so the dataset card is the authority.

You can also bring your own CSV or JSONL with columns `text` and `label` (optional `script` with values `urdu`, `roman`, `mixed`, `unknown`) and set `data.path` in the config. Without a `script` column, the script is detected from the share of Arabic-script letters.

## Method

- **Data.** Rows with missing or very short text are dropped, text is normalised (Unicode NFKC, Arabic and Urdu yeh/kaf/heh unified, diacritics and invisible direction marks removed, Latin lowercased), and exact duplicate texts are removed so identical sentences cannot sit in both train and test. A report of dropped rows goes into `run_info.json`.
- **Script detection.** Share of Arabic-script letters among all letters: at least 0.7 is `urdu`, at most 0.3 is `roman`, otherwise `mixed`. Roman Urdu cannot be told apart from English by this check, so the data source has to guarantee the language.
- **Split.** One stratified train/val/test split from a fixed seed, stratified on label and script when every group is big enough. Both models see the same three frames.
- **Baseline.** TF-IDF word 1-2 grams plus character n-grams (`char_wb`, 2-5), logistic regression or linear SVM. Regularisation C is chosen on the validation split.
- **Transformer.** `xlm-roberta-base` (or `bert-base-multilingual-cased`) with a classification head, plain PyTorch loop, AdamW with linear warmup and decay, mixed precision on CUDA, early stopping on validation macro-F1, best epoch restored, `max_length` from the config.
- **Metrics.** Accuracy, macro-F1, per-class F1, per-script F1 and confusion matrix on the test split. Macro-F1 gets a percentile bootstrap 95% CI. The difference transformer minus baseline uses a paired bootstrap over the same resampled test rows.
- **Error analysis.** `error_analysis.json` holds top confusions overall and per script, short versus long text (threshold in words from the config), and counts of rows only one model gets right. `errors.csv` lists the model's misclassified test rows, longest first.
- **What keeps the comparison fair.** Same split, same normalised text, and both models tune only on the validation split. The test split is scored once.

## Repository structure

```
configs/   default.yaml (XLM-R), mbert.yaml, smoke.yaml (synthetic + tiny model)
src/urdu_sentiment/
  data.py            loading, cleaning, script detection, split
  synthetic.py       synthetic template sentences (tests and smoke only)
  baseline.py        TF-IDF + linear model
  backbone.py        tokenizer/model loading interface, tiny random backbone
  transformer.py     fine-tuning loop and batched prediction
  train.py           full run, writes predictions and run_info.json
  evaluate.py        metrics, bootstrap CIs, results table
  error_analysis.py  confusions, length and script breakdowns, errors.csv
  predict.py         inference wrapper used by the app
app/       Gradio demo and theme
scripts/   download_data.py
tests/     fast CPU tests
results/   outputs from real runs
```

## Limitations

- No result exists yet. The pipeline is tested on synthetic data and a tiny random model only; the real pretrained weights and a real dataset have not been run in this repo's build environment.
- Quality depends entirely on the dataset you pick: label source (human versus star-rating heuristics), domain and size. Results do not transfer across domains.
- Per-script numbers can rest on few rows. Groups under 30 rows are marked as small samples in `metrics.json`.
- Roman Urdu support is only as good as the Roman Urdu data you provide. Script detection cannot separate it from English, and it labels code-mixed text as `mixed` only when the Arabic-script share falls between 0.3 and 0.7.
- Sarcasm, negation and code-mixing are likely failure modes; check `errors.csv` rather than assuming.
- One seed, one split. The bootstrap captures test-set sampling noise, not training variance. Repeat with other seeds before drawing strong conclusions.
- The linear SVM option reports softmax of margins, which is not a calibrated probability. Transformer probabilities are often over-confident.
- The app is a demo, not a tool for decisions about people.

## Reproducing on free Colab or Kaggle

1. Open a new Colab notebook and select a GPU runtime (Runtime, Change runtime type, T4 GPU). On Kaggle, enable a GPU accelerator and internet access.
2. Run in cells:

```bash
!git clone https://github.com/qasimrazamalik-coder/urdu-sentiment-xlmr.git
%cd urdu-sentiment-xlmr
!pip install -e .
!python -m pytest -q
!python scripts/download_data.py --hf-id <owner/name> --text-col <text> --label-col <label>
!python -m urdu_sentiment.train --config configs/default.yaml
!python -m urdu_sentiment.evaluate --config configs/default.yaml
!python -m urdu_sentiment.error_analysis --config configs/default.yaml
```

3. Download `results/` (and `checkpoints/` if you want the app locally).

Approximate runtime, depending on the assigned GPU and dataset size: the baseline takes seconds to a few minutes on CPU; fine-tuning on a free T4 takes from several minutes for a few thousand short texts to well over an hour for hundreds of thousands. If you run out of memory, lower `model.batch_size` or `model.max_length` in the config.

## License and citations

Code: MIT, see `LICENSE`. Datasets keep their own licenses; check the dataset card before use or redistribution.

- Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale", ACL 2020 (XLM-RoBERTa).
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", NAACL 2019 (multilingual BERT).
