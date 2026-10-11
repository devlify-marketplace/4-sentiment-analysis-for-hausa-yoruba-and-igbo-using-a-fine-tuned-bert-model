# Project Report

## Title

**Sentiment Analysis for Hausa, Yoruba and Igbo Using a Fine-Tuned BERT Model**

## Abstract

This project develops a multilingual three-class sentiment classifier for text in Hausa, Yoruba and Igbo. It fine-tunes an Africa-adapted XLM-RoBERTa encoder (`Davlan/afro-xlmr-base`) on the corresponding language partitions of NaijaSenti, a human-annotated Nigerian Twitter corpus. The training workflow uses publisher-defined train, validation and test partitions; validation macro-F1 selects the checkpoint, and the held-out test partition is reserved for final evaluation. The delivered CPU-sized run uses a deterministic, proportion-preserving sample of 1,000 training examples and 800 validation/test examples per language; this is explicitly recorded in the model metadata and results. The software supports full-split training by setting both caps to zero. The system reports accuracy, macro/weighted precision, recall and F1, confusion matrices and language-level scores, and exposes the model through a local API and browser interface. It is a reproducible academic prototype, not a deployed content-moderation or public-opinion system.

## 1. Introduction

Sentiment analysis systems are disproportionately developed and evaluated for high-resource languages. Hausa, Yoruba and Igbo have large speaker communities, but digital language resources and widely available task-specific systems are comparatively limited. A multilingual transformer adapted to African-language text provides a practical starting point for a supervised classification prototype.

This implementation operationalizes the selected topic as prediction of negative, neutral or positive polarity for a user-supplied short text, with a language code chosen from Hausa, Yoruba and Igbo. XLM-RoBERTa is a BERT-family masked-language transformer encoder and supports sequence-classification fine-tuning; the selected AfroXLMR checkpoint was further adapted to African languages including the three target languages.

## 2. Problem statement

Generic sentiment tools may not represent the vocabulary, spelling conventions, code-mixing, dialectal variation and contextual cues of Hausa, Yoruba and Igbo. The problem studied is whether a shared Africa-focused pretrained encoder, fine-tuned on human-labeled examples, can classify sentiment on held-out examples for each of these three languages. A single reported accuracy would conceal language-specific disparities, so the system reports macro-F1 and class-level results separately for each language.

## 3. Aim and objectives

**Aim:** design, implement and evaluate a reproducible transformer-based sentiment classifier for Hausa, Yoruba and Igbo text.

**Objectives:**
1. Acquire and validate an established human-annotated three-class sentiment dataset for each target language.
2. Fine-tune one shared AfroXLMR-base sequence-classification model on the combined multilingual training data.
3. Use official validation splits for model selection and keep official test splits separate until final evaluation.
4. Report accuracy, precision, recall, macro-F1, weighted-F1 and confusion matrices, overall and by language.
5. Provide a local API and browser demo for controlled qualitative inspection.
6. Document reproducibility, dataset provenance, licensing, ethics and scope limitations.

## 4. Research questions

1. How well does the shared fine-tuned model classify negative, neutral and positive posts across the three target languages on held-out NaijaSenti examples?
2. How consistent is macro-F1 across Hausa, Yoruba and Igbo, and which sentiment classes account for errors?
3. What limitations arise from short social-media posts, code-mixing, dataset domain, and low-resource spelling variation?

## 5. Scope

The project uses only Hausa, Yoruba and Igbo from NaijaSenti; Nigerian Pidgin is excluded. It performs sentence-level three-class text classification, not translation, emotion classification, aspect extraction, or sentiment-intensity regression. It does not infer the identity or beliefs of a writer, monitor live social media, or make enforcement decisions. It is an offline coursework prototype and local demo.

## 6. Data and preprocessing

NaijaSenti is a published human-annotated Twitter corpus with official train, development (`dev`) and test partitions. The downloader records example counts, source split names, retrieval time and SHA-256 hashes. The model input is the published `tweet` text, paired with its sentiment label. Labels are normalized to negative/neutral/positive. Text content is otherwise retained as supplied, with no language translation, stopword deletion or stemming; the pretrained tokenizer performs subword tokenization and truncation to a maximum of 96 tokens by default. The report records actual partition counts at runtime.

The delivered CPU run samples 1,000 training examples and 800 validation/test examples per language. Sampling is deterministic, stratified by label, and approximately preserves each source split's observed class proportions. A run with caps set to zero processes all official examples. A language code is retained for stratified reporting, not passed as a separate learned feature. No validation/test examples update model parameters. Dataset licensing is documented in `DATA_PROVENANCE.md`.

## 7. Model and training method

`Davlan/afro-xlmr-base` initializes an XLM-RoBERTa encoder adapted on 17 African and three high-resource languages, including Hausa, Igbo and Yoruba. A randomly initialized three-logit classification head maps the contextual representation to sentiment classes. The network is fine-tuned end-to-end with cross-entropy loss and AdamW. Defaults: seed 42, learning rate 2e-5, weight decay 0.01, batch size 8, maximum sequence length 96, two epochs and 6% warmup. The local config uses 1,000 train and 800 evaluation examples per language for CPU feasibility. Exact software versions, split counts, run configuration and selected epoch are saved with the model. To run on all rows, set `--max-train-per-language 0 --max-eval-per-language 0`.

At the end of each epoch, performance on the sampled (or complete) validation split is summarized with macro-F1. The checkpoint with the best validation macro-F1 is retained. The official test partition is evaluated only after selecting the checkpoint; if evaluation caps are set, the selected subset is documented. Do not tune settings against test results.

## 8. Evaluation

The primary metric is macro-F1 because class frequency can differ and every sentiment class contributes equally. Supporting measures are accuracy, macro precision/recall, weighted precision/recall/F1, per-class precision/recall/F1/support, and a 3-by-3 confusion matrix. These are computed overall and separately for Hausa, Yoruba and Igbo. Results and provenance are saved to `artifacts/test_metrics.json`, predictions to `artifacts/test_predictions.csv` (text omitted), plots to `artifacts/*.png`, and a human-readable summary to `docs/RESULTS.md`.

The delivered metrics are based on a stratified subset, not the complete official test set; subset estimates have higher sampling uncertainty and must be described as such. Any comparison across languages should also account for different corpus sizes and class distributions. Test scores are benchmark measurements, not claims of expected performance for private messages, news, speech, customer feedback, other domains, or current platform content.

## 9. System implementation

The implementation is a Python package with data validation, training, inference and FastAPI modules. The browser demo posts explicitly selected language and user-entered text to the local API and displays the predicted class and model confidence. The API does not persist requests. Tests validate label normalization, required data columns, language allow-list, API schemas and the honest unready response without downloading the large model.

## 10. Ethics, privacy and limitations

The dataset reflects Twitter posts, a particular period, collection/filtering procedures and annotator judgments. It is not representative of all speakers, dialects, regions or contexts. Code-mixed examples, transliteration, spelling variation, sarcasm and cultural context are difficult for automatic classifiers. A correct label cannot be guaranteed and the probability score may be miscalibrated. No demographic or dialect fairness conclusions can be made from the corpus without suitable subgroup labels and community-led review.

The demo processes submitted text in memory and does not write it to disk, but users should avoid entering sensitive personal information. The model must not be used for high-impact decisions, surveillance, political profiling or automated moderation without separate legal review, consent, human oversight, domain-specific validation and an appeal process. Use of the training dataset must comply with its non-commercial ShareAlike license and attribution requirements.

## 11. Conclusion and future work

This project delivers a reproducible multilingual sentiment-classification baseline for three Nigerian languages, including fine-tuning artifacts and a local demonstration interface. The delivered checkpoint is a CPU-sized subset run; full-split training is available in code but requires substantially more compute. Future work should include community-led review of annotation quality, domain and dialect evaluation, calibration and abstention, a properly licensed comparison against a non-transformer baseline, fresh consented test data, and further adaptation using ethically sourced unlabeled text. The local checkpoint should not be considered operational until full evaluation and external validation are complete.

## References

1. Muhammad, S. H. et al. (2022). *NaijaSenti: A Nigerian Twitter Sentiment Corpus for Multilingual Sentiment Analysis*. Proceedings of LREC 2022, 590–602. <https://aclanthology.org/2022.lrec-1.63/>.
2. Alabi, J. O. et al. (2022). *Adapting Pre-trained Language Models to African Languages via Multilingual Adaptive Fine-Tuning*. Proceedings of COLING 2022, 4336–4349. <https://aclanthology.org/2022.coling-1.382/>.
3. Hugging Face. *HausaNLP/NaijaSenti-Twitter dataset card*. <https://huggingface.co/datasets/HausaNLP/NaijaSenti-Twitter>.
4. Hugging Face. *Davlan/afro-xlmr-base model card*. <https://huggingface.co/Davlan/afro-xlmr-base>.
