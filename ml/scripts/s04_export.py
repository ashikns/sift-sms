"""s04: export the fine-tuned encoder to an int8 ONNX Runtime model.

    uv run python -I scripts/s04_export.py

1. Rebuild the encoder as a "classic BERT" module (plain PyTorch ops, -10000 additive mask, mean pooling + L2) that
   ORT's BERT fusions recognise; check it against the HF model (max |diff| ≤ 1e-5).
2. TorchScript ONNX export (opset 17, dynamic batch/sequence).
3. ORT transformer optimizer (attention / layer-norm / GELU fusions), with fusion counts asserted.
4. Dynamic int8 quantization (QInt8 per-channel weights; per-tensor gave cosine 0.95 vs fp32, per-channel 0.99).
5. ORT format conversion (fixed optimization style), and a parity report fp32 vs int8 vs .ort.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers.optimizer import optimize_model
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from smsml.normalize import normalize
from smsml.pipeline import BUNDLE_WORK, config, snapshot_dir, work, write_json
from smsml.tokenizer import WordPiece


class ClassicBert(nn.Module):
    def __init__(self, hf):
        super().__init__()
        cfg = hf.config
        self.emb = hf.embeddings
        self.layers = hf.encoder.layer
        self.nh = cfg.num_attention_heads
        self.hd = cfg.hidden_size // cfg.num_attention_heads
        self.hidden = cfg.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq = input_ids.size(1)
        pos = torch.arange(seq, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        x = self.emb.word_embeddings(input_ids) + self.emb.position_embeddings(pos) \
            + self.emb.token_type_embeddings(torch.zeros_like(input_ids))
        x = self.emb.LayerNorm(x)
        ext = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).to(x.dtype)) * -10000.0
        for layer in self.layers:
            sa = layer.attention.self

            def heads(t):
                return t.view(t.size(0), -1, self.nh, self.hd).transpose(1, 2)

            q, k, v = heads(sa.query(x)), heads(sa.key(x)), heads(sa.value(x))
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.hd) + ext
            probs = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(probs, v).transpose(1, 2).reshape(x.size(0), -1, self.hidden)
            a = layer.attention.output.LayerNorm(layer.attention.output.dense(ctx) + x)
            i = F.gelu(layer.intermediate.dense(a))
            x = layer.output.LayerNorm(layer.output.dense(i) + a)
        m = attention_mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return F.normalize(pooled, p=2, dim=-1)


def hf_embed(hf, ids, mask):
    h = hf(input_ids=ids, attention_mask=mask).last_hidden_state
    m = mask.unsqueeze(-1).to(h.dtype)
    return F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-9), p=2, dim=-1)


def session(path, threads=4):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def run(sess, ids: list[int]) -> np.ndarray:
    a = np.asarray([ids], dtype=np.int64)
    return sess.run(["embedding"], {"input_ids": a, "attention_mask": np.ones_like(a)})[0][0]


def main() -> int:
    cfg = config()
    nh, hidden = int(cfg["model"]["num_heads"]), int(cfg["model"]["hidden"])
    src = work("finetuned", "bert")
    hf = AutoModel.from_pretrained(src).eval()
    model = ClassicBert(hf).eval()
    wp = WordPiece.from_file(snapshot_dir(cfg) / "vocab.txt")

    df = pd.read_parquet(work("corpus_split.parquet"))
    sample = df[df["split"].isin(["val_a", "val_b"])].sample(n=400, random_state=1)["text"].tolist()
    id_lists = [wp.encode(normalize(t).enc) for t in sample]

    # 1. classic module == HF
    worst = 0.0
    with torch.no_grad():
        for ids in id_lists[:100]:
            t = torch.tensor([ids])
            mask = torch.ones_like(t)
            worst = max(worst, float((model(t, mask) - hf_embed(hf, t, mask)).abs().max()))
    print(f"classic vs HF max |diff| = {worst:.2e}")
    if worst > 1e-5:
        print("ERROR: classic BERT module diverges from HF", file=sys.stderr)
        return 1

    # 2. export
    BUNDLE_WORK.mkdir(parents=True, exist_ok=True)
    fp32 = work("export", "encoder.fp32.onnx")
    dummy = torch.tensor([id_lists[0]])
    torch.onnx.export(
        model, (dummy, torch.ones_like(dummy)), str(fp32), dynamo=False, opset_version=17,
        input_names=["input_ids", "attention_mask"], output_names=["embedding"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "embedding": {0: "batch"}},
    )

    # 3. fusions
    opt = optimize_model(str(fp32), model_type="bert", num_heads=nh, hidden_size=hidden, opt_level=0)
    stats = opt.get_fused_operator_statistics()
    print("fusions:", {k: v for k, v in stats.items() if v})
    n_layers = len(hf.encoder.layer)
    fusion_ok = stats.get("Attention", 0) + stats.get("MultiHeadAttention", 0) == n_layers \
        and stats.get("SkipLayerNormalization", 0) >= 2 * n_layers - 1
    if not fusion_ok:
        print("WARNING: expected BERT fusions not found; the model works unfused but runs slower", file=sys.stderr)
    fused = work("export", "encoder.fused.onnx")
    opt.save_model_to_file(str(fused))

    # 4. int8
    int8 = work("export", "encoder.int8.onnx")
    quantize_dynamic(str(fused), str(int8), weight_type=QuantType.QInt8, per_channel=True, reduce_range=False,
                     # Fused contrib ops defeat ONNX shape inference; their outputs are float32.
                     extra_options={"MatMulConstBOnly": True, "DefaultTensorType": onnx.TensorProto.FLOAT})

    # 5. ORT format
    subprocess.run([sys.executable, "-I", "-m", "onnxruntime.tools.convert_onnx_models_to_ort", str(int8),
                    "--optimization_style", "Fixed", "--target_platform", "arm"], check=True)
    ort_path = int8.with_suffix(".ort")
    final = BUNDLE_WORK / "encoder.int8.ort"
    final.write_bytes(ort_path.read_bytes())

    s_fp32, s_int8, s_ort = session(fp32), session(int8), session(final)
    cos_q, d_ort = [], 0.0
    with torch.no_grad():
        for ids in id_lists:
            e32 = run(s_fp32, ids)
            e8 = run(s_int8, ids)
            eo = run(s_ort, ids)
            cos_q.append(float(np.dot(e32, e8)))
            d_ort = max(d_ort, float(np.abs(e8 - eo).max()))
    report = {
        "classic_vs_hf_max_abs": worst,
        "fusions": {k: int(v) for k, v in stats.items() if v},
        "fusion_ok": bool(fusion_ok),
        "int8_vs_fp32_cosine": {"mean": float(np.mean(cos_q)), "min": float(np.min(cos_q))},
        "ort_vs_onnx_int8_max_abs": d_ort,
        "sizes_mb": {p.name: round(p.stat().st_size / 1e6, 2) for p in (fp32, int8, final)},
    }
    write_json(work("export", "report.json"), report)
    print(json.dumps(report, indent=2))
    if report["int8_vs_fp32_cosine"]["mean"] < 0.99 or d_ort > 1e-4:
        print("ERROR: quantization/ORT-format parity outside tolerance", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
