# modified from  https://github.com/FireRedTeam/FireRedASR2S/blob/main/fireredasr2s/fireredasr2/asr.py
# add vad processing
import logging
import os
import re
import tempfile
import time
import traceback
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Optional, BinaryIO
import numpy as np

import torch

try:
    import soundfile as sf
except ImportError:
    sf = None
try:
    import librosa
except ImportError:
    librosa = None

from .data.asr_feat import ASRFeatExtractor
from .models.fireredasr_aed import FireRedAsrAed
from .models.fireredasr_llm import FireRedAsrLlm
from .models.lstm_lm import LstmLm
from .models.param import count_model_parameters
from .tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer
from .tokenizer.llm_tokenizer import LlmTokenizerWrapper

from whosaidwhat.pipe.fireredasr.models.fireredasr_with_vad import decode_audio, format_timestamp
from whosaidwhat.pipe.fireredasr.models.vad import (
    SpeechTimestampsMap,
    VadOptions,
    collect_chunks,
    get_speech_timestamps,
)

logger = logging.getLogger(__name__)

# Target sample rate for ASR (must match feat extractor / training).
_SAMPLING_RATE = 16000


def _load_audio_int16_scale(wav_path: str) -> np.ndarray:
    """Load audio as float32 in int16 scale [-32768, 32767], 16 kHz mono.
    Matches test_fireredasr2_with_vad.py / load_audio_first_channel_16k so that
    ASR features match training and test script.
    """
    if sf is None or librosa is None:
        # Fallback: use decode_audio and scale to int16 range
        audio = decode_audio(wav_path, sampling_rate=_SAMPLING_RATE)
        return (audio * 32768.0).astype(np.float32)
    audio, sr = sf.read(wav_path, dtype="int16")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != _SAMPLING_RATE:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=_SAMPLING_RATE)
    return audio.astype(np.float32)


@dataclass
class FireRedAsr2Config:
    use_gpu: bool = True
    use_half: bool = False
    beam_size: int = 3
    nbest: int = 1
    decode_max_len: int = 0
    softmax_smoothing: float = 1.25
    aed_length_penalty: float = 0.6
    eos_penalty: float = 1.0
    return_timestamp: bool = False
    decode_min_len: bool = 0
    repetition_penalty: float = 1.0
    llm_length_penalty: float = 0.0
    temperature: float = 1.0
    elm_dir: str = ""
    elm_weight: float = 0.0
    def __post_init__(self):
        pass


class FireRedAsr2:
    @classmethod
    def from_pretrained(cls, asr_type, model_dir, config=FireRedAsr2Config()):
        assert asr_type in ["aed", "llm"]

        cmvn_path = os.path.join(model_dir, "cmvn.ark")
        feat_extractor = ASRFeatExtractor(cmvn_path)

        if asr_type == "aed":
            model_path = os.path.join(model_dir, "model.pth.tar")
            dict_path =os.path.join(model_dir, "dict.txt")
            spm_model = os.path.join(model_dir, "train_bpe1000.model")
            model = load_fireredasr_aed_model(model_path)
            tokenizer = ChineseCharEnglishSpmTokenizer(dict_path, spm_model)
        elif asr_type == "llm":
            model_path = os.path.join(model_dir, "model.pth.tar")
            encoder_path = os.path.join(model_dir, "asr_encoder.pth.tar")
            llm_dir = os.path.join(model_dir, "Qwen2-7B-Instruct")
            model, tokenizer = load_firered_llm_model_and_tokenizer(
                model_path, encoder_path, llm_dir)
        elm = None
        if config.elm_dir:
            assert os.path.exists(config.elm_dir), f"{config.elm_dir}"
            model_path = os.path.join(config.elm_dir, "model.pth.tar")
            elm = load_lstm_lm(model_path)
            elm.eval()
            logger.info(elm)
        count_model_parameters(model)
        model.eval()
        return cls(asr_type, feat_extractor, model, tokenizer, elm, config)

    def __init__(self, asr_type, feat_extractor, model, tokenizer, elm, config):
        self.asr_type = asr_type
        self.feat_extractor = feat_extractor
        self.model = model
        self.tokenizer = tokenizer
        self.elm = elm
        self.config = config
        logger.info(self.config)
        if self.config.use_gpu:
            if self.config.use_half:
                self.model.half()
            self.model.cuda()
            if self.elm:
                self.elm.cuda()
        else:
            self.model.cpu()
    def _apply_vad_filter_from_audio(
        self,
        audio_int16_scale: np.ndarray,
        audio_normalized: np.ndarray,
        vad_parameters: Optional[VadOptions] = None,
        chunk_length: float = 30.0,
    ) -> Tuple[List[np.ndarray], List[Dict[str, float]], List[Dict[str, int]]]:
        """Run VAD on normalized audio; slice chunks from int16-scale audio for ASR."""
        duration = audio_int16_scale.shape[0] / _SAMPLING_RATE
        logging.info(f"Processing audio with duration {format_timestamp(duration)}")

        if duration < chunk_length:
            clip_timestamps = [{"start": 0, "end": audio_int16_scale.shape[0]}]
        else:
            if vad_parameters is None:
                vad_parameters = VadOptions(
                    max_speech_duration_s=chunk_length,
                    min_silence_duration_ms=160,
                )
            clip_timestamps = get_speech_timestamps(
                audio_normalized, vad_parameters, sampling_rate=_SAMPLING_RATE
            )

        audio_chunks, chunks_metadata = collect_chunks(
            audio_int16_scale, clip_timestamps, sampling_rate=_SAMPLING_RATE, max_duration=chunk_length
        )

        duration_after_vad = (
            sum((s["end"] - s["start"]) for s in clip_timestamps) / _SAMPLING_RATE
        )
        logging.info(f"VAD filter removed {format_timestamp(duration - duration_after_vad)} of audio")
        return audio_chunks, chunks_metadata, clip_timestamps

    def apply_vad_filter(
        self,
        audio: np.ndarray,
        vad_parameters: Optional[VadOptions] = None,
        chunk_length: float = 30.0
    ) -> Tuple[List[np.ndarray], List[Dict[str, float]], List[Dict[str, int]]]:
        """
        Apply VAD filter to audio and return chunks with metadata.
        (Kept for compatibility; prefers _apply_vad_filter_from_audio for int16-scale ASR.)
        """
        return self._apply_vad_filter_from_audio(
            audio, audio.copy(), vad_parameters, chunk_length
        )
    def transcribe_with_vad(
        self,
        batch_uttid: List[str],
        batch_wav_path: List[str],
        vad_parameters: Optional[VadOptions] = None,
        chunk_length: float = 30.0
    ) -> List[Dict]:
        """
        Transcribe audio with VAD filtering for long audio support.

        Args:
            batch_uttid: List of utterance IDs
            batch_wav_path: List of audio file paths
            vad_parameters: VAD parameters
            chunk_length: Maximum chunk length in seconds

        Returns:
            List of transcription results
        """
        all_results = []

        for uttid, wav_path in zip(batch_uttid, batch_wav_path):
            # Load audio in int16-scale float32 (same as test_fireredasr2_with_vad.py)
            audio = _load_audio_int16_scale(wav_path)
            # VAD expects normalized float; use normalized copy for timestamps only
            audio_vad = (audio / 32768.0).astype(np.float32)
            # Apply VAD filter on normalized audio; chunks will be cut from original int16-scale
            audio_chunks, chunks_metadata, clip_timestamps = self._apply_vad_filter_from_audio(
                audio, audio_vad, vad_parameters, chunk_length
            )
            logging.info(f"wav_path: {wav_path}, after apply vad filter: audio_chunks: {audio_chunks}, chunks_metadata: {chunks_metadata}, clip_timestamps: {clip_timestamps}")
            if not audio_chunks:
                # No speech detected
                all_results.append({
                    "uttid": uttid,
                    "text": "",
                    "wav": wav_path,
                    "rtf": "0.0000"
                })
                continue

            # Process each chunk
            chunk_results = []
            total_duration = 0
            start_time = time.time()

            for i, (chunk, metadata) in enumerate(zip(audio_chunks, chunks_metadata)):
                if len(chunk) == 0:
                    continue

                # Use temp wav file so feat_extractor uses kaldiio.load_mat(path), matching
                # the file-input path and avoiding any tuple-path handling differences.
                # Chunk is int16-scale float32; write as int16 PCM.
                temp_wav_path = None
                try:
                    fd, temp_wav_path = tempfile.mkstemp(suffix=".wav", prefix="asr_chunk_")
                    os.close(fd)
                    chunk_int16 = np.clip(chunk, -32768, 32767).astype(np.int16)
                    sf.write(temp_wav_path, chunk_int16, _SAMPLING_RATE)
                    chunk_result = self.transcribe([uttid], [temp_wav_path])
                finally:
                    if temp_wav_path and os.path.exists(temp_wav_path):
                        try:
                            os.remove(temp_wav_path)
                        except OSError:
                            pass

                if chunk_result:
                    chunk_results.extend(chunk_result)
                logging.info(f"chunk_index: {i}, chunk_result: {chunk_result}")

                total_duration += metadata["duration"]

            # Combine results
            if chunk_results:
                # Combine all text from chunks
                combined_text = " ".join([result["text"] for result in chunk_results])

                # Calculate RTF
                elapsed = time.time() - start_time
                rtf = elapsed / total_duration if total_duration > 0 else 0

                all_results.append({
                    "uttid": uttid,
                    "text": combined_text,
                    "wav": wav_path,
                    "rtf": f"{rtf:.4f}",
                    "chunks": len(audio_chunks),
                    "vad_timestamps": clip_timestamps
                })
            else:
                all_results.append({
                    "uttid": uttid,
                    "text": "",
                    "wav": wav_path,
                    "rtf": "0.0000"
                })

        return all_results

    @torch.no_grad()
    def transcribe(self, batch_uttid, batch_wav_path, use_vad=False, vad_parameters=None, chunk_length=30.0):
        if use_vad:
            return self.transcribe_with_vad(batch_uttid, batch_wav_path, vad_parameters, chunk_length)

        # Original transcribe logic without VAD
        batch_uttid_origin = batch_uttid
        try:
            feats, lengths, durs, batch_wav_path, batch_uttid = \
                self.feat_extractor(batch_wav_path, batch_uttid)
            if feats is None:
                return [{"uttid": uttid, "text":""} for uttid in batch_uttid_origin]
        except:
            traceback.print_exc()
            return [{"uttid": uttid, "text":""} for uttid in batch_uttid_origin]
        total_dur = sum(durs)
        if self.config.use_gpu:
            feats, lengths = feats.cuda(), lengths.cuda()
            if self.config.use_half:
                feats = feats.half()

        if self.asr_type == "aed":
            start_time = time.time()

            try:
                hyps = self.model.transcribe(
                    feats, lengths,
                    self.config.beam_size,
                    self.config.nbest,
                    self.config.decode_max_len,
                    self.config.softmax_smoothing,
                    self.config.aed_length_penalty,
                    self.config.eos_penalty,
                    self.config.return_timestamp,
                    self.elm,
                    self.config.elm_weight
                )
            except Exception as e:
                traceback.print_exc()
                hyps = []

            elapsed = time.time() - start_time
            rtf= elapsed / total_dur if total_dur > 0 else 0

            results = []
            for uttid, wav, hyp, dur in zip(batch_uttid, batch_wav_path, hyps, durs):
                hyp = hyp[0]  # only return 1-best
                hyp_ids = [int(id) for id in hyp["yseq"].cpu()]
                text = self.tokenizer.detokenize(hyp_ids)
                text = re.sub(r"(<blank>)|(<sil>)", "", text)
                results.append({"uttid": uttid, "text": text.lower(),
                    "confidence": round(hyp["confidence"].cpu().item(), 3),
                    "dur_s": round(dur, 3), "rtf": f"{rtf:.4f}"})
                if type(wav) == str:
                    results[-1]["wav"] = wav
                if self.config.return_timestamp:
                    results[-1]["timestamp"] = self._get_and_fix_timestamp(hyp, hyp_ids, dur)
            return results

        elif self.asr_type == "llm":
            input_ids, attention_mask, _, _ = \
                LlmTokenizerWrapper.preprocess_texts(
                    origin_texts=[""]*feats.size(0), tokenizer=self.tokenizer,
                    max_len=128, decode=True)
            if self.config.use_gpu:
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
            start_time = time.time()

            try:
                generated_ids = self.model.transcribe(
                    feats, lengths, input_ids, attention_mask,
                    self.config.beam_size,
                    self.config.decode_max_len,
                    self.config.decode_min_len,
                    self.config.repetition_penalty,
                    self.config.llm_length_penalty,
                    self.config.temperature
                )
                texts = self.tokenizer.batch_decode(generated_ids,
                                                    skip_special_tokens=True)
            except Exception as e:
                texts = []

            elapsed = time.time() - start_time
            rtf= elapsed / total_dur if total_dur > 0 else 0
            results = []
            for uttid, wav, text in zip(batch_uttid, batch_wav_path, texts):
                results.append({"uttid": uttid, "text": text.lower(),
                                "rtf": f"{rtf:.4f}"})
                if type(wav) == str:
                    results[-1]["wav"] = wav
            return results

    def _get_and_fix_timestamp(self, hyp, hyp_ids, dur):
        r3 = lambda x: round(x, 3)
        if "timestamp" not in hyp or hyp["timestamp"] is None:
            timestamp = []
            avg_dur = dur / len(hyp_ids) if len(hyp_ids) > 0 else 0
            last_end = dur
            for i, hyp_id in enumerate(hyp_ids):
                token = self.tokenizer.detokenize([hyp_id], "", False)
                start = min(max(0, i*avg_dur), last_end)
                end = min((i+1)*avg_dur, dur)
                last_end = end
                timestamp.append([token.lower(), r3(start), r3(end)])
        else:
            starts, ends = hyp["timestamp"]
            timestamp = []
            last_end = dur
            SHIFT = 0.06  # shift 40ms
            for hyp_id, start, end in zip(hyp_ids, starts, ends):
                token = self.tokenizer.detokenize([hyp_id], "", False)
                start = min(max(0, start - SHIFT), last_end)
                end = min(max(0, end - SHIFT), dur)
                last_end = end
                timestamp.append([token.lower(), r3(start), r3(end)])
        # Fix case: start == dur and end == dur
        for i in range(len(timestamp)):
            idx = -(i+1)
            _, start, end = timestamp[idx]
            if abs(dur - start) < 0.001:
                logger.info(f"start before {timestamp[idx]}")
                timestamp[idx][1] = dur - (i+1)*0.001
                logger.info(f"start after {timestamp[idx]}")
            if i != 0 and abs(dur - end) < 0.001:
                logger.info(f"end before {timestamp[idx]}")
                timestamp[idx][2] = dur - i*0.001
                logger.info(f"end after {timestamp[idx]}")
        timestamp = self.tokenizer.merge_spm_timestamp(timestamp)
        return timestamp


def load_fireredasr_aed_model(model_path):
    package = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    model = FireRedAsrAed.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=False)
    return model


def load_firered_llm_model_and_tokenizer(model_path, encoder_path, llm_dir):
    package = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    package["args"].encoder_path = encoder_path
    package["args"].llm_dir = llm_dir
    model = FireRedAsrLlm.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=False)
    tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(llm_dir)
    return model, tokenizer

def load_lstm_lm(model_path):
    package = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    model = LstmLm.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=False)
    return model
