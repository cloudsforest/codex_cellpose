# Cellpose Streamlit Fine-Tuning App

A Streamlit app for loading microscopy image sets, previewing default Cellpose segmentation overlays, and fine-tuning a Cellpose model on custom masks.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Naming format

Use the filename regex box to parse sample IDs and channels, for example:

```regex
(?P<sample>.+?)(?:_(?P<channel>ch\d+))?
```

- `sample` groups files into a sample set.
- `channel` is optional and used for display ordering.

## Fine-tuning

Upload matched image/mask files where image and mask filenames map to the same `sample` value.
