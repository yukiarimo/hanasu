import torch
from pathlib import Path
from typing import Optional
import utils
from models import SynthesizerTrn
from text import symbols
import numpy as np
import onnxruntime
from scipy.io.wavfile import write
from data_utils import get_text

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
    posterior_channels = 128

    model_g = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )

    _ = model_g.eval()
    _ = utils.load_checkpoint(model_path, model_g, None)

    def infer_forward(text, text_lengths, scales, sid=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        audio = model_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
        )[0]

        return audio

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

    phoneme_ids = get_text(text, hps)
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