import torch
from pathlib import Path
from typing import Optional
from hanasu import commons, utils
from hanasu.models import load_model
from hanasu.text import symbols
import numpy as np
import onnxruntime
from scipy.io.wavfile import write
from hanasu.data_utils import get_text

def export_onnx(model_path: str, config_path: str, output: str) -> None:
    """
    Export model to ONNX format.

    Args:
        model_path: Path to model weights (.pth)
        config_path: Path to model config (.json)
        output: Path to output model (.onnx)
    """
    torch.manual_seed(1234)
    model_path = Path(model_path)
    config_path = Path(config_path)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    hps = utils.get_hparams_from_file(config_path)
    model_g = load_model(config_path, model_path, 'cpu')

    def infer_forward(text, text_lengths, scales, sid=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]

        # Replicating SynthesizerTrn.infer logic without autocast
        if model_g.n_speakers > 0:
            g = model_g.emb_g(sid).unsqueeze(-1)
        else:
            g = None

        x, m_p, logs_p, x_mask = model_g.enc_p(text, text_lengths, g=g)
        logw = model_g.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        w = torch.exp(logw) * x_mask * length_scale
        w = torch.clamp(w, min=0.1)
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        noise = torch.randn_like(m_p)
        z_p = m_p + noise * torch.exp(logs_p) * noise_scale
        z = model_g.flow(z_p, y_mask, g=g, reverse=True)
        z_masked = z * y_mask
        o = model_g.dec(z_masked, g=g)
        return o

    model_g.forward = infer_forward

    dummy_input_length = 50
    sequences = torch.randint(
        low=0, high=len(symbols), size=(1, dummy_input_length), dtype=torch.long
    )
    sequence_lengths = torch.LongTensor([sequences.size(1)])

    sid: Optional[torch.LongTensor] = None
    if hps.data.n_speakers > 1:
        sid = torch.LongTensor([0])

    # noise, length, noise_w
    scales = torch.FloatTensor([0.667, 1.0, 0.8])
    dummy_input = (sequences, sequence_lengths, scales, sid)

    # Export
    torch.onnx.export(
        model=model_g,
        args=dummy_input,
        f=str(output),
        verbose=False,
        opset_version=15,
        input_names=["input", "input_lengths", "scales", "sid"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 1: "time1", 2: "time2"},
        },
    )

    print(f"Exported model to {output}")

def synthesize(
    model_path,
    config_path,
    output_wav_path,
    text,
    sid=None,
    scales=None
):
    sess_options = onnxruntime.SessionOptions()
    model = onnxruntime.InferenceSession(str(model_path), sess_options=sess_options, providers=["CPUExecutionProvider"])
    hps = utils.get_hparams_from_file(config_path)

    phoneme_ids = get_text(text)
    text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
    text_lengths = np.array([text.shape[1]], dtype=np.int64)

    if scales is None:
        scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)

    sid_np = np.array([int(sid)]) if sid is not None else None

    audio = model.run(
        None,
        {
            "input": text,
            "input_lengths": text_lengths,
            "scales": scales,
            "sid": sid_np,
        },
    )[0].squeeze((0, 1))

    write(data=audio, rate=hps.data.sampling_rate, filename=output_wav_path)
    return audio