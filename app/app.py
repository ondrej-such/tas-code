"""
app.py -- a small web application that turns your own recordings into
ECAPA-TDNN speaker embeddings, in the format of data/ecapa_embeddings.csv.

Start it with
    python app.py
and open the address it prints (http://127.0.0.1:7860). When the app runs on
your own computer, the recordings never leave it. The first computation
downloads the ECAPA-TDNN model (about 80 MB).
"""

import os
import tempfile

import gradio as gr
import pandas as pd

from embed import compute_embeddings, write_csv

INTRO = """
# Speaker embeddings from your own recordings

1. Type a **speaker** label and, optionally, a **story** label (what was read).
2. Upload a recording or record one with the microphone, then press **Add recording**.
   Repeat for every speaker; two or more speakers are needed, three as in the paper.
3. Press **Compute embeddings** and download the CSV.

Each recording is cut into short clips (400 ms in the paper); each clip becomes a
192-dimensional embedding computed by the ECAPA-TDNN network. The first part of every
recording is used for training and the rest for testing. The CSV can be read by the
notebooks in place of `data/ecapa_embeddings.csv`.

Read for at least 30 seconds per speaker, in a quiet room. Accepted formats: WAV, FLAC, OGG, MP3.
"""

COLS = ["speaker", "story", "recording"]


def add_recording(recs, speaker, story, audio):
    speaker = (speaker or "").strip()
    story = (story or "").strip() or "1"
    if not speaker:
        raise gr.Error("Please type a speaker label.")
    if not audio:
        raise gr.Error("Please upload or record audio first.")
    if any(c in speaker + story for c in ',"'):
        raise gr.Error("Labels may not contain commas or quotes.")
    n = sum(r["speaker"] == speaker and r["story"] == story for r in recs) + 1
    name = f"{speaker}_{story}_{n}"
    recs = recs + [dict(speaker=speaker, story=story, path=audio, name=name)]
    return recs, table(recs), None


def table(recs):
    return pd.DataFrame([{k: r[k if k != "recording" else "name"] for k in COLS} for r in recs], columns=COLS)


def clear(_):
    return [], table([]), None, None, None


def run(recs, seg_ms, overlap, train_pct, balance, progress=gr.Progress()):
    if len({r["speaker"] for r in recs}) < 2:
        raise gr.Error("Add recordings of at least two speakers.")
    try:
        rows, summary = compute_embeddings(
            recs, seg_ms=int(seg_ms), overlap=overlap / 100,
            train_fraction=train_pct / 100, balance=balance,
            progress=lambda f, m: progress(f, desc=m))
    except ValueError as e:
        raise gr.Error(str(e))
    out = os.path.join(tempfile.mkdtemp(), "my_embeddings.csv")
    write_csv(rows, out)
    counts = pd.DataFrame([{k: v for k, v in r.items() if k != "emb"} for r in rows])
    by = [c for c in ("speaker", "story", "set") if c in counts]
    counts = counts.groupby(by).size().reset_index(name="clips")
    return out, pd.DataFrame(summary), counts


with gr.Blocks(title="Speaker embeddings") as demo:
    gr.Markdown(INTRO)
    recs = gr.State([])
    with gr.Row():
        with gr.Column(scale=1):
            speaker = gr.Textbox(label="Speaker", placeholder="e.g. A")
            story = gr.Textbox(label="Story (optional)", placeholder="e.g. EN")
        with gr.Column(scale=2):
            audio = gr.Audio(label="Recording", sources=["upload", "microphone"], type="filepath")
    with gr.Row():
        add_btn = gr.Button("Add recording", variant="secondary")
        clear_btn = gr.Button("Remove all recordings")
    rec_table = gr.Dataframe(value=table([]), label="Recordings", interactive=False)

    with gr.Accordion("Settings", open=False):
        seg_ms = gr.Slider(200, 2000, value=400, step=50, label="Clip length (ms)")
        overlap = gr.Slider(0, 75, value=0, step=5, label="Overlap between consecutive clips (%)")
        train_pct = gr.Slider(0, 90, value=70, step=5,
                              label="Training part of each recording (%); 0 = no train/test split")
        balance = gr.Checkbox(value=True, label="Give every speaker the same number of clips (per story)")

    run_btn = gr.Button("Compute embeddings", variant="primary")
    out_file = gr.File(label="Embeddings (CSV)")
    summary = gr.Dataframe(label="Recordings: length, share of speech, clips kept", interactive=False)
    counts = gr.Dataframe(label="Clips in the CSV", interactive=False)

    add_btn.click(add_recording, [recs, speaker, story, audio], [recs, rec_table, audio])
    clear_btn.click(clear, [recs], [recs, rec_table, out_file, summary, counts])
    run_btn.click(run, [recs, seg_ms, overlap, train_pct, balance], [out_file, summary, counts])

if __name__ == "__main__":
    demo.launch()
