# Preparing Release Archives

`examples/package_release.py` inventories the release files and builds two
archives under `dist/`:

- `dist/harnessadapt-code.tar.gz`: documentation, inference runtime, `H1` and
  `H2` harnesses, and model metadata files.
- `dist/harnessadapt-update45.tar`: the complete `update45` Hugging Face model
  export (`models/update45/`), including the four `safetensors` weight shards.

It copies actual file bytes into the archives, rejects symlinks, and excludes
cached files and output directories. Archive entries use relative paths and
reproducible ownership/timestamp metadata.

```bash
python examples/package_release.py
```

The code archive includes the model's text configuration, model card, tokenizer,
weight index, and license. The model archive includes the complete export with
the four weight shards (`~19.3 GB`). Extract both into the same directory;
shared metadata files are identical.

`FILE_MANIFEST.json` records the relative path, byte size, and SHA-256 digest of
every file in the release, and `dist/SHA256SUMS` records the SHA-256 digests of
the completed archives.
