# Data provenance and responsible use

## Primary benchmark

- **Dataset:** NaijaSenti Twitter sentiment corpus, Version 1.0.
- **Dataset card:** [HausaNLP/NaijaSenti-Twitter](https://huggingface.co/datasets/HausaNLP/NaijaSenti-Twitter).
- **Original project and files:** [hausaNLP/NaijaSenti](https://github.com/hausanlp/NaijaSenti), publisher-maintained `data/annotated_tweets/{hau,ibo,yor}/{train,dev,test}.tsv`.
- **Paper:** Muhammad et al. (2022), [LREC 2022 proceedings](https://aclanthology.org/2022.lrec-1.63/), pp. 590–602.
- **Languages used:** Hausa (`hau`), Igbo (`ibo`) and Yoruba (`yor`). Nigerian Pidgin is intentionally excluded to match the selected project topic.
- **Task:** three-way sentiment classification (`negative`, `neutral`, `positive`).
- **Source modality/domain:** human-annotated Twitter posts. The dataset card says @mentions were anonymized as `@user` and URLs removed.
- **Splits:** use the publisher's predefined train, development (`dev`), and test partitions without recombining or random re-splitting. In local files, `dev` is named `validation` for compatibility with the training workflow. Train updates parameters, validation selects a checkpoint, and test is evaluated once at the end.
- **Retrieval method:** `scripts/download_data.py` downloads official raw TSV files from `raw.githubusercontent.com/hausanlp/NaijaSenti/main` and converts them to UTF-8 CSV. It does not execute remote dataset-loader Python code. `data/raw/manifest.json` stores source URL, raw-file SHA-256, converted-file SHA-256, counts and retrieval timestamp.

The Hugging Face dataset card lists 14,172/2,677/5,303 Hausa train/validation/test examples; 8,522/2,090/4,515 Yoruba; and 10,192/1,841/3,682 Igbo. These are reference counts: the downloader logs and manifests retrieved counts and raises on unexpected schema/split issues rather than fabricating replacements.

## Licensing

The Hugging Face dataset card identifies NaijaSenti as **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**. The GitHub project homepage separately displays CC BY 4.0. In the absence of a definitive reconciliation, this project follows the more restrictive dataset-card statement: attribute the dataset, do not use it commercially, and share adaptations under the same terms where required. Verify current licensing with the curators before redistribution or any use that may be commercial. The model checkpoint has a separate MIT license; project code is MIT licensed. These licenses are not interchangeable.

The downloader gets the dataset files from the publisher's repository, does not scrape Twitter, and records SHA-256 hashes and counts in `data/raw/manifest.json`. Local raw files are produced only when the downloader is run and should be handled under the dataset license. The repository does not manufacture, augment or silently substitute training examples.

## Limitations

The corpus is not a probability sample of people in Nigeria, covers one platform and collection period, and reflects the annotators' definitions and judgments. It is not guaranteed to be exclusively monolingual; code-mixed examples and community-specific usage are part of the benchmark. Label imbalance and short informal text can affect scores. Systematic evaluation by dialect, script, age, geography or code-mixing requires appropriately governed data and human review.
