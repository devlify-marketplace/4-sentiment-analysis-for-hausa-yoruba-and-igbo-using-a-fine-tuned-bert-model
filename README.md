# Sentiment Analysis for Hausa, Yoruba and Igbo Using a Fine-Tuned BERT Model

**Topic 4** from the supplied final-year-project list. This project fine-tunes an Africa-adapted XLM-RoBERTa encoder (a BERT-family transformer encoder) for three-way sentiment classification of Hausa, Yoruba and Igbo text. It includes an authentic public dataset downloader, reproducible train/validation/test workflow, per-language evaluation, a saved model, a local REST API, browser demo, tests, and an academic report.

> **Scope and interpretation.** The benchmark is NaijaSenti Version 1.0: annotated Twitter posts collected for research, not a representative sample of all Nigerian opinions or all domains. The supplied local checkpoint is explicitly subset-trained because full end-to-end training exceeded the available CPU run window; `artifacts/model_metadata.json` records the exact caps. The code supports full-data training on a faster machine. Do not describe the subset metrics as full-test-set performance.

## What is included

- `scripts/download_data.py`: downloads the publisher's raw official TSV files for Hausa (`hau`), Igbo (`ibo`) and Yoruba (`yor`) from the NaijaSenti GitHub repository, converts the publisher's `dev` partition to `validation`, and writes checksums and a manifest. It executes no remote dataset code.
- `src/naija_sentiment/train.py`: seeded fine-tuning of `Davlan/afro-xlmr-base` with validation-only checkpoint selection and a held-out test evaluation.
- `src/naija_sentiment/predict.py`: local batch prediction API for individual texts.
- `src/naija_sentiment/api.py` and `web/index.html`: FastAPI backend and responsive demo.
- `tests/`: unit/API tests that do not download data or a model.
- `docs/`: research report, model card, provenance and generated evaluation results.
- `configs/default.json`: 2 epochs, 1,000 train rows/language, and 800 validation/test rows/language for a CPU-sized default run.

## Requirements

Python 3.10–3.12, internet access for the first dataset/model download, and about 5 GB free disk space for the Python environment, cached pretrained weights, data and fine-tuned checkpoint. Training works on CPU; an NVIDIA GPU is optional and substantially faster.

```bash
cd /workspace/topic-04-sentiment-analysis-hausa-yoruba-igbo
python3 -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e '.[test]'
```

Alternatively, `pip install -r requirements.txt && pip install -e .` installs the same project package.

## Download data

```bash
# To refresh the bundled dataset files:
python scripts/download_data.py --root . --overwrite
```

The project bundle already includes all nine official split CSVs and `data/raw/manifest.json`; normally skip this refresh step. When run, the downloader reads the publisher's official `train.tsv`, `dev.tsv` and `test.tsv` files directly from [hausanlp/NaijaSenti](https://github.com/hausanlp/NaijaSenti), saving them as `data/raw/` CSVs. It records the raw-file and converted-file SHA-256 hashes, dataset ID/version, split counts, retrieval time, and source URLs. This avoids executing a remote Python dataset script. No synthetic data are generated. See [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md) before using or redistributing the data.

## Fine-tune and evaluate

```bash
python -m naija_sentiment.train --root .
```

The default configuration uses a deterministic **class-stratified sample that preserves each split's observed sentiment proportions**: 1,000 training examples and 800 validation/test examples per language. This makes the checked-in local checkpoint practical to recreate on CPU; metrics are subset estimates. The official train split is used for parameter updates, the validation split selects the checkpoint by macro-F1, and the selected model is evaluated on the sampled test split.

To use every row in every official split, run:

```bash
python -m naija_sentiment.train --root . --max-train-per-language 0 --max-eval-per-language 0
```

The full-data run is compute-intensive on CPU and is best run with a suitable GPU or larger compute window. The test set is not used for checkpoint selection. Every run records sample caps, split counts, seed, device and hyperparameters in `artifacts/model_metadata.json`.

Outputs are written to `artifacts/`:

- `model/`: tokenizer, fine-tuned weights and config.
- `model_metadata.json`, `training_history.csv`, `test_metrics.json`, `test_predictions.csv`.
- `confusion_matrix.png`, `language_metrics.png` and `training_history.png`.

## Run the API and demo

After training has created `artifacts/model/`:

```bash
uvicorn naija_sentiment.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Interactive API docs: <http://127.0.0.1:8000/docs>.

- `GET /api/health` — readiness and model identity.
- `GET /api/languages` — supported language codes and labels.
- `POST /api/predict` — one text and one language code.
- `POST /api/predict-batch` — up to 100 texts in one language.

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"Na gode sosai!","language":"hau"}'
```

The service is local, does not persist submitted text, and is not hardened for public deployment. Predictions are estimates, not facts about authors or their beliefs.

## Tests

```bash
pytest -q
```

Tests cover label normalization, required data columns, language validation, proportion-preserving sampling and API readiness without requiring internet or model weights.

## Dataset and model citations

- Muhammad et al. (2022), [NaijaSenti: A Nigerian Twitter Sentiment Corpus for Multilingual Sentiment Analysis](https://aclanthology.org/2022.lrec-1.63/), LREC 2022, pp. 590–602. Cite this work in any report, presentation, or publication using the corpus.
- Alabi et al. (2022), [Adapting Pre-trained Language Models to African Languages via Multilingual Adaptive Fine-Tuning](https://aclanthology.org/2022.coling-1.382/), COLING 2022. The `Davlan/afro-xlmr-base` checkpoint is based on XLM-R and reports adaptation to Hausa, Igbo and Yoruba.
- [Dataset card and licensing](https://huggingface.co/datasets/HausaNLP/NaijaSenti-Twitter).

Dataset licensing is separate from the code and pretrained model. The Hugging Face dataset card identifies NaijaSenti as **CC BY-NC-SA 4.0**; the GitHub project page has also displayed CC BY 4.0. This project conservatively follows the dataset-card statement pending clarification from the curators. Not for commercial use without permission. Code is MIT licensed; the pretrained checkpoint has its own MIT model license. See `docs/DATA_PROVENANCE.md`.
