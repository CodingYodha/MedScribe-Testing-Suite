# robust_diarize_and_label.py
import os
import numpy as np
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.signal import medfilt
from faster_whisper import WhisperModel, BatchedInferencePipeline
from math import ceil

# -------- CONFIG --------
MODEL_DIR = "pretrained_models/spkrec-ecapa-voxceleb"   # local ECAPA folder
AUDIO_PATH = r"E:\Projects\Med_Scribe\Testing\audio.wav"  # local audio (wav or m4a if ffmpeg backend)
WHISPER_MODEL_PATH = "./models/large-v3"                 # your faster-whisper model folder
DEVICE = "cuda"                                         # or "cpu"
WINDOW_SEC = 30.0        # longer context to reduce pitch-sensitivity
OVERLAP_SEC = 4.0       # 50% overlap
MAX_SPEAKERS = 5
SIM_THRESHOLD = 0.80   # cosine similarity threshold for reassignment
MEDIAN_KERNEL_SEC = 3.0 # smoothing kernel in seconds (odd integer multiple of frames)
# ------------------------

def load_audio(path):
    sig, sr = torchaudio.load(path)
    # ensure mono
    if sig.shape[0] > 1:
        sig = sig.mean(dim=0, keepdim=True)
    return sig, sr

def chunk_audio(signal, sr, window_sec, overlap_sec):
    win = int(window_sec * sr)
    stride = int((window_sec - overlap_sec) * sr)
    total = signal.shape[1]
    starts = list(range(0, max(1, total - win + 1), stride))
    if starts[-1] + win < total:
        starts.append(max(0, total - win))
    chunks = []
    times = []
    for s in starts:
        e = min(s + win, total)
        chunks.append(signal[:, s:e])
        times.append((s / sr, e / sr))
    return chunks, times

def get_embeddings(classifier, chunks):
    embs = []
    for chunk in chunks:
        # classifier.encode_batch accepts waveform like (batch, time) or (channels, time) as you used earlier
        with torch.no_grad():
            emb = classifier.encode_batch(chunk)  # returns tensor-like
        # normalize & flatten to vector
        emb = torch.tensor(emb).squeeze().cpu().numpy()
        emb = emb.reshape(-1)  # ensure 1D
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        embs.append(emb)
    return np.vstack(embs)  # shape: (n_chunks, emb_dim)

def estimate_num_speakers(embeddings, max_k=5):
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import AgglomerativeClustering
    best_k, best_score = 2, -1

    n_samples = len(embeddings)
    max_k = min(max_k, n_samples - 1)  # ✅ prevent invalid k

    for k in range(2, max_k + 1):
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k

def cluster_and_reassign(embeddings, sim_threshold, max_k):
    # estimate K
    k = estimate_num_speakers(embeddings, max_k)
    if k == 1:
        labels = np.zeros(len(embeddings), dtype=int)
    else:
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(embeddings)

    # compute means
    unique = sorted(set(labels))
    means = {u: embeddings[labels == u].mean(axis=0) for u in unique}
    for u in means:
        means[u] = means[u] / (np.linalg.norm(means[u]) + 1e-8)

    # reassignment by cosine similarity to means (stability hack)
    for i, emb in enumerate(embeddings):
        sims = {u: float(np.dot(emb, means[u])) for u in unique}
        best_u, best_sim = max(sims.items(), key=lambda x: x[1])
        if best_sim >= sim_threshold:
            labels[i] = best_u
        # else keep original cluster (rare)
    return labels, k

def labels_to_segments(labels, times):
    segs = []
    cur_label = labels[0]
    cur_start = times[0][0]
    for i in range(1, len(labels)):
        if labels[i] != cur_label:
            cur_end = times[i][0]  # end is start of current chunk
            segs.append((cur_start, cur_end, cur_label))
            cur_label = labels[i]
            cur_start = times[i][0]
    # finish
    segs.append((cur_start, times[-1][1], cur_label))
    return segs

def median_smooth_labels(labels, times, kernel_sec):
    # map labels to per-chunk sequence and apply medfilt with odd kernel
    if kernel_sec <= 0:
        return labels
    avg_chunk_dur = (times[0][1] - times[0][0])
    kernel = int(round(kernel_sec / avg_chunk_dur))
    if kernel % 2 == 0:
        kernel += 1
    if kernel < 1:
        kernel = 1
    if kernel == 1:
        return labels
    return medfilt(labels.astype(float), kernel_size=kernel).astype(int)

def align_transcript_with_speakers(trans_segments, diarization_segments):
    # diarization_segments: list of (start,end,label)
    out = []
    for t in trans_segments:
        s, e, text = t.start, t.end, t.text.strip()
        # find diarization segment with max overlap
        best_label, best_overlap = None, 0.0
        for ds in diarization_segments:
            ds_s, ds_e, ds_label = ds
            overlap = max(0.0, min(e, ds_e) - max(s, ds_s))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = ds_label
        out.append((best_label, s, e, text))
    return out

def main():
    print("Loading models...")
    classifier = EncoderClassifier.from_hparams(source=MODEL_DIR, savedir=None, run_opts={"device": DEVICE})
    whisper_model = WhisperModel(WHISPER_MODEL_PATH, device=DEVICE, compute_type="float16", download_root="./models/whisper")
    batched_model = BatchedInferencePipeline(model=whisper_model)
    print("Models loaded.")

    signal, sr = load_audio(AUDIO_PATH)
    print(f"Audio loaded: {AUDIO_PATH} sr={sr} duration={signal.shape[1]/sr:.2f}s")

    # chunk audio
    chunks, times = chunk_audio(signal, sr, WINDOW_SEC, OVERLAP_SEC)
    print(f"{len(chunks)} chunks, window={WINDOW_SEC}s overlap={OVERLAP_SEC}s")

    # embeddings
    embeddings = get_embeddings(classifier, chunks)
    print("Embeddings computed:", embeddings.shape)

    # clustering + reassignment
    labels, detected_k = cluster_and_reassign(embeddings, SIM_THRESHOLD, MAX_SPEAKERS)
    print(f"Detected speakers (initial/after): {detected_k}, labels unique: {sorted(set(labels))}")

    # smoothing
    labels_sm = median_smooth_labels(labels, times, MEDIAN_KERNEL_SEC)
    diarization_segments = labels_to_segments(labels_sm, times)
    print("Diarization segments (merged):")
    for s,e,l in diarization_segments:
        print(f"  {s:.2f}s - {e:.2f}s : Speaker {l+1}")

    # Transcribe with faster-whisper BatchedInferencePipeline
    print("Transcribing with faster-whisper...")
    segments_gen, info = batched_model.transcribe(AUDIO_PATH, batch_size=16, language='en')
    segments = list(segments_gen)
    print(f"Transcribed {len(segments)} segments")
    transcript_chunks = [segment.text.strip() for segment in segments if segment.text.strip()]
    full_transcript = " ".join(transcript_chunks)

    print(50*"=" , '\n')
    print(full_transcript)
    print(50*"=", '\n')
    # Align transcript sentences with diarization segments
    labeled = align_transcript_with_speakers(segments, diarization_segments)
    print("\n===== SPEAKER-LABELED TRANSCRIPT =====\n")
    for label, s, e, text in labeled:
        if label is None:
            print(f"Unknown Speaker: {text}")
        else:
            print(f"Speaker {label+1}: {text}")
    print("\nDone.")

    # -----------------------------
    # Entity extraction using Gemma
    # -----------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
    import torch

    # Configure 4-bit quantization exactly as before
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model_dir = "./models/gemma-3-1b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    gemma_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    print("Gemma loaded")

    # Stopping criteria for your AAA delimiter
    class StopOnBackticks(StoppingCriteria):
        def __init__(self, tokenizer, stop_sequence="AAA"):
            self.tokenizer = tokenizer
            self.stop_sequence = stop_sequence
            self.stop_ids = tokenizer.encode(stop_sequence, add_special_tokens=False)

        def __call__(self, input_ids, scores, **kwargs):
            if len(input_ids[0]) >= len(self.stop_ids):
                if (input_ids[0][-len(self.stop_ids):] == torch.tensor(self.stop_ids, device=input_ids.device)).all():
                    return True
            return False

    stopping_criteria = StoppingCriteriaList([StopOnBackticks(tokenizer)])

    # -----------------------------
    # Prepare system + user prompt
    # -----------------------------
    system_prompt = """
    You are a medical prescription parser. Extract ONLY information explicitly stated.

    Rules:
    1. Extract medicines with EXACT dosages mentioned
    2. If dosage/frequency unclear, mark as "unspecified"
    3. Do NOT infer or assume any information
    4. If doctor says "continue previous meds", extract NOTHING
    5. Output only one valid JSON object and stop
    6. At the end of the Output print AAA
    7. Your task is to extract medicines , diseases, tests

    Output format:
    {
    "medicines": [{"name": str, "dosage": str, "frequency": str, "duration": str}],
    "diseases": [str],
    "tests": [{"name": str, "timing": str}]
    }
    """

    user_prompt = f"""
    Extract from this prescription conversation:
    {full_transcript}
    Remember: Only extract explicitly stated information. No assumptions.
    """

    # -----------------------------
    # Run Gemma model
    # -----------------------------
    inputs = tokenizer(system_prompt, return_tensors='pt').to(gemma_model.device)
    user_inputs = tokenizer(user_prompt, return_tensors='pt').to(gemma_model.device)

    input_ids = torch.cat([inputs.input_ids, user_inputs.input_ids], dim=1)
    attention_mask = torch.cat([inputs.attention_mask, user_inputs.attention_mask], dim=1)

    outputs = gemma_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=512,
        temperature=0.01,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping_criteria,
        do_sample=False
    )

    result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if result_text.startswith(system_prompt):
        result_text = result_text[len(system_prompt):].strip()
    print(result_text)


if __name__ == "__main__":
    main()

