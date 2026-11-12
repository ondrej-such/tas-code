"""
embed.py -- speaker embeddings from your own recordings.

Same pipeline as the one used for data/ecapa_embeddings.csv in the paper:

  * mono mix, resample to 16 kHz
  * energy voice-activity detection on 25 ms frames with a 10 ms hop: a frame
    is speech if it is within 40 dB of the loudest frame of the recording
  * optional split of each recording in time: the part holding the first
    TRAIN_FRACTION of its speech frames is training audio, the rest is test
    audio; clips never cross the boundary
  * clips of SEG_MS milliseconds on a fixed grid with stride
    SEG_MS * (1 - OVERLAP); a clip is kept if at least 80% of its frames are
    speech
  * optionally, within each story (and set) all speakers are balanced to the
    same number of clips by evenly subsampling the longer ones
  * ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb) -> 192-dimensional
    embedding, normalised to unit length

Output: one CSV row per clip with columns
    speaker, story, recording, set, start_ms, dim_001 ... dim_192
(the column "set" is present only when the recordings are split).

Command line:
    python embed.py out.csv A=alice.wav B=bob.wav A/story2=alice2.wav
Each argument is SPEAKER=FILE or SPEAKER/STORY=FILE.
"""

import argparse
import os
import warnings

import numpy as np
import soundfile as sf
import torch
import torchaudio

warnings.filterwarnings("ignore", category=UserWarning)

TARGET_SR = 16000
VAD_FLOOR_DB = 40.0
MIN_SPEECH_FRAC = 0.80
FR_LEN = int(TARGET_SR * 0.025)
FR_HOP = int(TARGET_SR * 0.010)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models", "ecapa")

_model = None


def load_model():
    """Load ECAPA-TDNN once (downloaded on first use, about 80 MB)."""
    global _model
    if _model is None:
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        _model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=MODEL_DIR,
            run_opts={"device": "cpu"},
            local_strategy=LocalStrategy.COPY,
        )
    return _model


def load_16k_mono(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(np.ascontiguousarray(wav.mean(axis=1))).unsqueeze(0)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0).numpy()


def speech_frames(sig):
    """True for 25 ms frames within VAD_FLOOR_DB of the loudest frame."""
    if len(sig) < FR_LEN:
        return np.zeros(0, dtype=bool)
    fr_idx = np.arange(0, len(sig) - FR_LEN + 1, FR_HOP)
    frames = sig[fr_idx[:, None] + np.arange(FR_LEN)[None, :]]
    fr_db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-12)
    return fr_db > (fr_db.max() - VAD_FLOOR_DB)


def cut_positions(is_speech, lo, hi, seg_len, stride):
    """Starts (in samples) of the kept clips on a fixed grid inside [lo, hi)."""
    cand = np.arange(lo, hi - seg_len + 1, stride)
    if len(cand) == 0 or len(is_speech) == 0:
        return cand[:0]
    csum = np.concatenate([[0], np.cumsum(is_speech)])
    f1 = np.minimum(cand // FR_HOP, len(is_speech) - 1)
    f2 = np.minimum((cand + seg_len - FR_LEN) // FR_HOP, len(is_speech) - 1)
    frac = (csum[f2 + 1] - csum[f1]) / (f2 - f1 + 1)
    return cand[frac >= MIN_SPEECH_FRAC]


def compute_embeddings(recordings, seg_ms=400, overlap=0.0, train_fraction=0.7,
                       balance=True, progress=None):
    """
    recordings: list of dicts with keys "speaker", "story", "path"
                (and optionally "name", used in the "recording" column).
    train_fraction: None or 0 for no split; otherwise in (0, 1).
    progress: optional callable progress(fraction, message).
    Returns (rows, summary): rows are dicts ready for write_csv; summary is a
    list of per-recording dicts (duration, share of speech, clips kept).
    """
    if not recordings:
        raise ValueError("No recordings.")
    split = bool(train_fraction) and 0 < train_fraction < 1
    seg_len = int(TARGET_SR * seg_ms / 1000)
    stride = max(1, int(round(TARGET_SR * seg_ms * (1 - overlap) / 1000)))
    say = progress or (lambda f, m: None)

    ## 1. Cut every recording.
    clips = []          # (rec index, set, start)
    sigs, summary = [], []
    for r, rec in enumerate(recordings):
        say(0.3 * r / len(recordings), f"Reading {rec.get('name', rec['path'])}")
        sig = load_16k_mono(rec["path"])
        is_speech = speech_frames(sig)
        sigs.append(sig)
        if split:
            n_sp = is_speech.sum()
            b = int(np.searchsorted(np.cumsum(is_speech), train_fraction * n_sp)) * FR_HOP
            parts = {"train": (0, b), "test": (b, len(sig))}
        else:
            parts = {"all": (0, len(sig))}
        kept = {}
        for st, (lo, hi) in parts.items():
            starts = cut_positions(is_speech, lo, hi, seg_len, stride)
            kept[st] = len(starts)
            clips += [(r, st, s) for s in starts]
        summary.append(dict(recording=rec.get("name", os.path.basename(rec["path"])),
                            speaker=rec["speaker"], story=rec["story"],
                            seconds=round(len(sig) / TARGET_SR, 1),
                            speech_pct=round(100 * float(is_speech.mean()) if len(is_speech) else 0, 1),
                            **{f"clips_{k}": v for k, v in kept.items()}))

    ## 2. Balance speakers within each story (and set) by even subsampling.
    if balance:
        groups = {}
        for c in clips:
            rec = recordings[c[0]]
            groups.setdefault((rec["story"], c[1]), {}).setdefault(rec["speaker"], []).append(c)
        clips = []
        for by_spk in groups.values():
            n = min(len(v) for v in by_spk.values())
            for v in by_spk.values():
                pick = np.round(np.linspace(0, len(v) - 1, n)).astype(int) if n > 0 else []
                clips += [v[i] for i in pick]
    if not clips:
        raise ValueError("No clips: the recordings are too short or too quiet for this clip length.")

    ## 3. Embed in batches.
    model = load_model()
    rows = []
    batch = 64
    for b0 in range(0, len(clips), batch):
        say(0.3 + 0.7 * b0 / len(clips), f"Embedding clips {b0 + 1}-{min(b0 + batch, len(clips))} of {len(clips)}")
        part = clips[b0:b0 + batch]
        segs = np.stack([sigs[r][s:s + seg_len] for r, _, s in part]).astype(np.float32)
        with torch.no_grad():
            emb = model.encode_batch(torch.from_numpy(segs)).squeeze(1).numpy()
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        for (r, st, s), e in zip(part, emb):
            rec = recordings[r]
            row = dict(speaker=rec["speaker"], story=rec["story"],
                       recording=rec.get("name", os.path.basename(rec["path"])))
            if split:
                row["set"] = st
            row["start_ms"] = int(round(1000 * s / TARGET_SR))
            row["emb"] = e
            rows.append(row)
    rows.sort(key=lambda x: (x["speaker"], x["story"], x["recording"], x.get("set", ""), x["start_ms"]))
    say(1.0, "Done")
    return rows, summary


def write_csv(rows, path):
    meta = [k for k in ("speaker", "story", "recording", "set", "start_ms") if k in rows[0]]
    header = meta + [f"dim_{i:03d}" for i in range(1, len(rows[0]["emb"]) + 1)]
    with open(path, "w", newline="") as f:
        f.write(",".join(f'"{h}"' for h in header) + "\n")
        for r in rows:
            vals = [f'"{r[k]}"' if k != "start_ms" else str(r[k]) for k in meta]
            f.write(",".join(vals + [f"{v:.9g}" for v in r["emb"]]) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output CSV")
    ap.add_argument("recordings", nargs="+", help="SPEAKER=FILE or SPEAKER/STORY=FILE")
    ap.add_argument("--seg-ms", type=int, default=400)
    ap.add_argument("--overlap", type=float, default=0.0)
    ap.add_argument("--train-fraction", type=float, default=0.7, help="0 for no split")
    ap.add_argument("--no-balance", action="store_true")
    a = ap.parse_args()
    recs = []
    for item in a.recordings:
        label, path = item.split("=", 1)
        spk, story = (label.split("/", 1) + ["1"])[:2]
        recs.append(dict(speaker=spk, story=story, path=path, name=os.path.basename(path)))
    rows, summary = compute_embeddings(recs, a.seg_ms, a.overlap, a.train_fraction,
                                       not a.no_balance, lambda f, m: print(m))
    for s in summary:
        print(s)
    write_csv(rows, a.out)
    print(f"wrote {a.out}: {len(rows)} clips")


if __name__ == "__main__":
    main()
