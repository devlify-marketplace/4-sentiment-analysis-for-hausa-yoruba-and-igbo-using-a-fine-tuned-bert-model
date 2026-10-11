# Model Card — Hausa, Yoruba and Igbo Sentiment Classifier

## Model identity

This is a fine-tuned three-class text classifier built from [`Davlan/afro-xlmr-base`](https://huggingface.co/Davlan/afro-xlmr-base), an XLM-RoBERTa-base encoder adapted with multilingual adaptive fine-tuning for African languages, including Hausa, Igbo and Yoruba. XLM-R is a BERT-family encoder with a sequence-classification head added and fine-tuned for the downstream task. The actual checkpoint and run metadata are written to `artifacts/model/` and `artifacts/model_metadata.json`.

## Intended use

Coursework, research demonstration, and qualitative inspection of short text resembling the NaijaSenti annotated Twitter benchmark. Outputs are categorical estimates (`negative`, `neutral`, `positive`) with model probabilities. Language code is supplied by the caller and recorded in output; the model does not independently verify language identity.

## Out-of-scope use

Do not use this model for consequential decisions, surveillance, political profiling, hiring, credit, education, health, law enforcement, automated content removal, or as evidence of an author's beliefs or intent. It is not a general opinion poll or replacement for human interpretation, moderation policy or community review. Confidence values have not been calibrated and must not be interpreted as guaranteed probabilities of correctness.

## Training data and procedure

The model is fine-tuned on NaijaSenti Version 1.0 Hausa (`hau`), Yoruba (`yor`) and Igbo (`ibo`) text. It uses the publisher's train partition for gradient updates, validation partition for best-checkpoint selection by macro-F1, and test partition for final evaluation after selection. The delivered CPU-sized run uses deterministic, class-stratified samples of 1,000 training and 800 validation/test examples per language, approximately preserving each split's observed label proportions. Its caps and counts are recorded in the artifact metadata. To fine-tune on all official rows, pass `--max-train-per-language 0 --max-eval-per-language 0`. Preprocessing uses the published text and label, canonicalizes sentiment labels, and applies the AfroXLMR tokenizer with a maximum length of 96 tokens. No generated examples are used. See `DATA_PROVENANCE.md` for source and license details.

## Evaluation

Evaluation reports accuracy, macro/weighted precision, recall and F1, per-class scores, support, and confusion matrices overall and by language. Results are in `RESULTS.md` after the training run. The official test partition is not used for model selection. Delivered metrics use the recorded test subset and have greater sampling uncertainty than complete-split evaluation. A test score does not establish domain generalization or fairness. Validation gaps include dialect, code-mixing, orthography, script variation, sarcasm, regional context, changing vocabulary and non-Twitter text.

## Ethical and privacy considerations

NaijaSenti consists of social-media posts and should be used under the dataset's terms. The dataset card states CC BY-NC-SA 4.0; treat the data and any derivatives accordingly unless the dataset curators clarify otherwise. The local demo does not persist requests, but users should avoid entering sensitive personal text. Retain human judgment, disclose use, and avoid stigmatizing or profiling language communities.

## Licenses

The pretrained model card identifies the base checkpoint as MIT licensed. Project code is MIT licensed. The benchmark data has a distinct license (see `DATA_PROVENANCE.md`); the code/model license does not grant additional rights to the dataset.
