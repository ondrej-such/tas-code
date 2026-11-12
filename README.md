# tas-code

Code accompanying the paper *Multiplicative miscalibration of binary logistic models*.

## Contents

| File | Description |
|---|---|
| `figures.ipynb` | Recreates the eleven figures of the paper (R kernel). |
| `supplement.ipynb` | Supplementary analyses (R kernel). S1: robustness of Section 5 to the train/test split. |
| `data/ecapa_embeddings.csv` | 192-dimensional ECAPA-TDNN embeddings of 400 ms clips of three speakers, used in Figures 7–11. |
| `app/` | Web application (Python, Gradio) that computes embeddings from your own recordings in the same format. |

The audio recordings from which the embeddings were computed are not distributed.

## Running the notebooks

The notebooks need R with the [IRkernel](https://irkernel.github.io/) Jupyter kernel and the packages `logistf`, `arm`, `glmnet` and `MASS`. Start Jupyter in this folder so that `data/` is found. `figures.ipynb` runs in under a minute and `supplement.ipynb` in about two minutes.

## The embeddings file

One row per clip. Columns:

- `speaker`: A, M or S
- `story`: the text read (EN or EB)
- `set`: `train` (first 70% of each recording) or `test` (last 30%)
- `start_ms`: start of the clip within the recording, in milliseconds
- `dim_001` … `dim_192`: the embedding

## Embeddings from your own recordings

`app/` contains a small web application that turns recordings into ECAPA-TDNN embeddings in the format of `data/ecapa_embeddings.csv`, so that the notebooks can be rerun on new speakers (for example, students in a class). Recordings can be uploaded or made with the microphone in the browser. When the application runs on your own computer, the recordings never leave it.

It needs Python 3.10–3.12:

```bash
cd app
python -m venv .venv
.venv/bin/pip install -r requirements.txt     # on Windows: .venv\Scripts\pip
.venv/bin/python app.py                       # then open http://127.0.0.1:7860
```

The first computation downloads the ECAPA-TDNN model (about 80 MB). The same pipeline is available from the command line:

```bash
python embed.py my_embeddings.csv A=alice.wav B=bob.wav C=carol.wav
```

The CSV has the columns of `data/ecapa_embeddings.csv` plus a `recording` column. With the default settings (400 ms clips, no overlap, the first 70% of each recording for training, speakers balanced within each story), the pipeline is the one used in the paper.
