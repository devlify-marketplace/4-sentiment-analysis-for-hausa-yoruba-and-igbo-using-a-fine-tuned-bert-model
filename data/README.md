# Data directory

`raw/` is populated by `python scripts/download_data.py --root .` from the publisher-maintained `HausaNLP/NaijaSenti-Twitter` dataset. The downloader stores official train/validation/test CSV splits for `hau`, `yor`, and `ibo`, plus a manifest with source, counts and file hashes. No synthetic rows are included. Dataset access and reuse are governed by the source dataset license; read `../docs/DATA_PROVENANCE.md` before reuse or redistribution.
