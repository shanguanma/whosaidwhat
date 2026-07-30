# copy and modified from https://github.com/MahmoudAshraf97/whisper-diarization/blob/main/diarize.py
import argparse
import logging
import os
import re
import glob
import sys
from pathlib import Path

import faster_whisper
import torch

from deepmultilingualpunctuation import PunctuationModel

from whosaidwhat.pipe.ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from whosaidwhat.pipe.fireredasr.models.fireredasr_with_vad import FireRedAsr, VadOptions
from whosaidwhat.pipe.fireredasr2.asr_with_vad import FireRedAsr2Config, FireRedAsr2
from whosaidwhat.pipe.helpers import (
    cleanup,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)

mtypes = {"cpu": "int8", "cuda": "float16"}

pid = os.getpid()
temp_outputs_dir = f"temp_outputs_{pid}"
temp_path = os.path.join(os.getcwd(), "temp_outputs")
os.makedirs(temp_path, exist_ok=True)

def parse_arguments():
    """Parse command line arguments."""
    # Initialize parser
    parser = argparse.ArgumentParser()
    #parser.add_argument(
    #    "-a", "--audio", help="name of the target audio file", required=True
    #)
    # Input/Output arguments
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Path to folder containing WAV files to process",
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        default="./output",
        help="Path to output folder for results",
    )
    parser.add_argument(
        "--skip-existing",
        type=str2bool,
        default=True,
        help="If True, skip audio files that already have output (.txt) in output-folder (resume mode). "
        "Set to False to re-decode all files.",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="*.wav",
        help="File pattern to match WAV files (e.g., '*.wav', 'recording_*.wav')",
    )
    parser.add_argument(
        "--no-stem",
        action="store_false",
        dest="stemming",
        default=True,
        help="Disables source separation."
        "This helps with long files that don't contain a lot of music.",
    )

    parser.add_argument(
        "--suppress_numerals",
        action="store_true",
        dest="suppress_numerals",
        default=False,
        help="Suppresses Numerical Digits."
        "This helps the diarization accuracy but converts all digits into written text.",
    )

    parser.add_argument(
        "--model-dir-or-name",
        default="/maduo/model_hub/FireRedASR-LLM-L",
        help="asr model dir",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        default=8,
        help="Batch size for batched inference, reduce if you run out of memory, "
        "set to 0 for original whisper longform inference",
    )

    parser.add_argument(
        "--language",
        type=str,
        default="chi", # Chinese language for ctc aligner normalize text
        help="Language spoken in the audio, specify None to perform language detection",
    )

    parser.add_argument(
        "--device",
        dest="device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="if you have a GPU use 'cuda', otherwise 'cpu'",
    )

    parser.add_argument(
        "--diarizer",
        default="msdd",
        choices=["msdd","streaming_softformer","offline_softformer","pyannote31"],
        help="Choose the diarization model to use",
    )
    parser.add_argument("--msdd-model-path", type=str, default="diar_msdd_telephonic", help="if it is name str , it will load nvidia nemo pretrained model, if it is model path, it will load a local model")
    parser.add_argument("--use-speaker-model-from-ckpt", type=str2bool, default=True, help="")
    parser.add_argument("--realign-with-punc", type=str2bool, default=True, help="Use the punctuation alignment model to realign speaker and txt")
    parser.add_argument("--punct-model", type=str,default="kredor/punctuate-all", help="it is str or local model path")

    parser.add_argument("--align-model", type=str, default="MahmoudAshraf/mms-300m-1130-forced-aligner",
            help="""choice ssl ctc model from huggingface,i.e.:https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py""")
    parser.add_argument("--local-files-only", type=str2bool, default=False, help="if local_files_only=True, args.model_name(asr model) is local directory.")
    parser.add_argument("--ctc-model-local-files-only", type=str2bool, default=False, help="if ctc_model_local_files_only=True, args.align_model is local directory.")

    # VAD related arguments
    parser.add_argument("--use-vad", type=str2bool, default=False, help="Enable VAD for long audio processing")
    parser.add_argument("--vad-chunk-length", type=float, default=30.0, help="Maximum chunk length in seconds for VAD")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="VAD speech threshold")
    parser.add_argument("--vad-min-speech-duration-ms", type=int, default=0, help="Minimum speech duration in milliseconds")
    parser.add_argument("--vad-min-silence-duration-ms", type=int, default=160, help="Minimum silence duration in milliseconds")
    parser.add_argument("--vad-speech-pad-ms", type=int, default=400, help="Speech padding in milliseconds")

    # pyannote31 related arguments
    parser.add_argument("--hf-token",type=str, help="using your huggingface token for loading pyannote-3.1 model")
    parser.add_argument("--max-speakers", type=int, default=4, help="If you know the maximum number of speakers a single audio file can contain in your test dataset, then set it to that number.")

    # firered_llm-l or fireredasr2-llm decoding
    parser.add_argument("--repetition-penalty", type=float,default=3.0, help="firered-llm-l default is 3.0, fireredasr2-llm default is 1.0")
    parser.add_argument("--llm-length-penalty", type=float,default=1.0, help="firered-llm-l default is 1.0, fireredasr2-llm default is 0.0")
    parser.add_argument("--temperature", type=float,default=1.0, help="firered-llm-l default is 1.0, fireredasr2-llm default is 1.0")
    args = parser.parse_args()
    return args



def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def find_wav_files(input_folder, file_pattern):
    """Find all WAV files in the input folder matching the pattern."""
    search_pattern = os.path.join(input_folder, file_pattern)
    wav_files = glob.glob(search_pattern)
    # remove resample.wav
    wav_files = [wav for wav in wav_files if "resampled" not in wav]
    if not wav_files:
        logging.info(
            f"No WAV files found matching pattern '{file_pattern}' in '{input_folder}'"
        )
        return []

    return sorted(wav_files)


def find_wav_files_from_jsonl(input_folder):
    import json
    wav_files=[]
    with open(f"{input_folder}/raw.jsonl",'r')as f:
        for line in f:
            d = json.loads(line.strip())
            wav_files.append(d["path"])
    return sorted(wav_files)


def get_completed_stems(output_folder):
    """
    Return the set of stem names that already have output in output_folder.
    Completion is determined by presence of {stem}.txt (same convention as process_single_audio).
    """
    completed = set()
    if not os.path.isdir(output_folder):
        return completed
    for name in os.listdir(output_folder):
        if name.endswith(".txt"):
            completed.add(Path(name).stem)
    return completed


def gen_txt_with_ws(args, audio_path, models):
    try:
        logging.info("generating asr output full_transcript...")
        asr_output_full_transcript = get_asr_output_without_timestamps(args, audio_path, models)

        logging.info("add timestampes to asr output full_transcript via pretrained CTC alignment model")
        word_timestamps = add_timestampes_using_alignment_model(args, audio_path, asr_output_full_transcript, models)
        logging.info(f"align model output word_timestamps: {word_timestamps}")
        # align model output word_timestamps: [{'start': 1.52, 'end': 1.78, 'text': 'hi', 'score': -2.4573097229003906}, {'start': 1.92, 'end': 2.0, 'text': 'my', 'score': -1.9522590637207031},...,{'start': 599.12, 'end': 599.34, 'text': 'twenty', 'score': -16.9932861328125}, {'start': 599.42, 'end': 599.6, 'text': 'twenty', 'score': -22.42353057861328},{'start': 599.68, 'end': 599.94, 'text': 'but', 'score': -6.4072265625}]

        logging.info("get speaker diarization output....")
        speaker_ts = gen_speaker_diarization_model_output(args, audio_path, models)
        logging.info(f"sd model output speaker_ts: {speaker_ts}") # [(start_ms, end_ms,speaker_index_num),..]
        # sd model output speaker_ts: [(720, 2080, 0), (2480, 4800, 1), (4640, 6640, 0), (6640, 7840, 1),...,(591120, 591520, 1), (592720, 595360, 1), (596640, 599990, 1),(599600, 599920, 0)]

        logging.info("words speaker mapping ....")
        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
        logging.info(f"get_words_speaker_mapping output: {wsm}")#start_time, end_time  are  millisecond
        # [{'word': 'hi', 'start_time': ms, 'end_time':ms,'speaker': num}]
        #get_words_speaker_mapping output:[{'word': 'hi', 'start_time': 1520, 'end_time': 1780, 'speaker': 0}, {'word': 'my', 'start_time': 1920, 'end_time': 2000, 'speaker': 0}, {'word': 'name', 'start_time': 2100, 'end_time': 2260, 'speaker': 0}, {'word': 'is', 'start_time': 2300, 'end_time': 2460, 'speaker': 0},...,{'word': 'twenty', 'start_time': 599120, 'end_time': 599340, 'speaker': 1}, {'word': 'twenty', 'start_time': 599420, 'end_time': 599600, 'speaker': 1},{'word': 'but', 'start_time': 599680, 'end_time': 599940, 'speaker': 1}]

        #return wsm, info, speaker_ts
        return wsm, speaker_ts
    except Exception as e:
        logging.error(f"Error in gen_txt_with_ws for {audio_path}: {str(e)}")
        logging.exception("Stack trace for gen_txt_with_ws:")
        logging.error("Exiting program due to model inference error.")
        sys.exit(1)

def get_asr_output_without_timestamps(args, audio_path, models):
    try:
        if "whisper" in args.model_dir_or_name:
            logging.info("using whisper model as ASR model ...")
            full_transcript = gen_whisper_asr_output(args, audio_path, models)
        elif "FireRedASR-LLM-L" in args.model_dir_or_name:
            logging.info("using FireRedASR-LLM-L model as ASR model ...")
            full_transcript = gen_firered_llm_l_asr_output(args, audio_path, models)
        elif "FireRedASR2-LLM" in args.model_dir_or_name:
            logging.info("using FireRedASR2-LLM model as ASR model ...")
            full_transcript = gen_fireredasr2_llm_asr_output(args, audio_path, models)
        return full_transcript
    except Exception as e:
        logging.error(f"Error in get_asr_output_without_timestamps for {audio_path}: {str(e)}")
        logging.error("Exiting program due to ASR model inference error.")
        sys.exit(1)

def gen_whisper_asr_output(args, audio_path, models):
    whisper_model = models.get('whisper_model')
    whisper_pipeline = models.get('whisper_pipeline')

    vocal_target = audio_path
    audio_waveform = faster_whisper.decode_audio(vocal_target) # float32 audio sample
    suppress_tokens = (
        find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
        if args.suppress_numerals
        else [-1]
    )

    # get language
    language = process_language_arg(args.language, args.model_dir_or_name)
    logging.info(f"specified {language}")

    if args.batch_size > 0:
        transcript_segments, info = whisper_pipeline.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            batch_size=args.batch_size,
        )
    else:
        transcript_segments, info = whisper_model.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            vad_filter=True,
        )

    full_transcript = "".join(segment.text for segment in transcript_segments)
    return full_transcript


def gen_fireredasr2_llm_asr_output(args, audio_path, models):
    audio_waveform = faster_whisper.decode_audio(audio_path) # float32 audio sample
    batch_uttid = [Path(audio_path).stem]
    batch_wav_path = [audio_path]
    model = models.get('fireredasr2_model')

    # Create VAD parameters if VAD is enabled
    vad_parameters = None
    if args.use_vad:
        vad_parameters = VadOptions(
            threshold=args.vad_threshold,
            min_speech_duration_ms=args.vad_min_speech_duration_ms,
            max_speech_duration_s=args.vad_chunk_length,
            min_silence_duration_ms=args.vad_min_silence_duration_ms,
            speech_pad_ms=args.vad_speech_pad_ms
        )

    results = model.transcribe(
        batch_uttid,
        batch_wav_path,
        use_vad=args.use_vad,
        vad_parameters=vad_parameters,
        chunk_length=args.vad_chunk_length
    )
    logging.info(results) # "uttid": uttid,
                   # "text": combined_text,
                   # "wav": wav_path,
                   # "rtf": f"{rtf:.4f}",
                   # "chunks": len(audio_chunks),
                   # "vad_timestamps": clip_timestamps
    full_transcript = results[0]['text']
    return full_transcript


def gen_firered_llm_l_asr_output(args, audio_path, models):
    audio_waveform = faster_whisper.decode_audio(audio_path) # float32 audio sample
    batch_uttid = [Path(audio_path).stem]
    batch_wav_path = [audio_path]
    model = models.get('firered_llm_l_model')

    # Create VAD parameters if VAD is enabled
    vad_parameters = None
    if args.use_vad:
        vad_parameters = VadOptions(
            threshold=args.vad_threshold,
            min_speech_duration_ms=args.vad_min_speech_duration_ms,
            max_speech_duration_s=args.vad_chunk_length,
            min_silence_duration_ms=args.vad_min_silence_duration_ms,
            speech_pad_ms=args.vad_speech_pad_ms
        )

    results = model.transcribe(
        batch_uttid,
        batch_wav_path,
        {
            "use_gpu": 1,
            "beam_size": 3,
            "decode_max_len": 0,
            "decode_min_len": 0,
            "repetition_penalty": args.repetition_penalty,
            "llm_length_penalty": args.llm_length_penalty,
            "temperature": args.temperature
        },
        use_vad=args.use_vad,
        vad_parameters=vad_parameters,
        chunk_length=args.vad_chunk_length
    )
    logging.info(results) # "uttid": uttid,
                   # "text": combined_text,
                   # "wav": wav_path,
                   # "rtf": f"{rtf:.4f}",
                   # "chunks": len(audio_chunks),
                   # "vad_timestamps": clip_timestamps
    full_transcript = results[0]['text']
    return full_transcript


def add_timestampes_using_alignment_model(args, audio_path, asr_output_full_transcript: str, models):
    try:
        # Forced Alignment
        logging.info(f"Forced Alignment....")
        # 为了稳定性，对齐模型按「每条音频」加载一次，用完立即释放。
        # 参考原始脚本，它们也是在函数内加载 alignment_model 然后 del。
        align_cfg = models.get("alignment_config")
        if align_cfg is None:
            align_device = args.device
            align_dtype = torch.float16 if args.device == "cuda" else torch.float32
            align_model_path = args.align_model
            align_local_only = args.ctc_model_local_files_only
        else:
            align_device = align_cfg["device"]
            align_dtype = align_cfg["dtype"]
            align_model_path = align_cfg["model_path"]
            align_local_only = align_cfg["local_files_only"]

        alignment_model, alignment_meta = load_alignment_model(
            align_device,
            model_path=align_model_path,
            dtype=align_dtype,
            local_files_only=align_local_only,
        )

        audio_waveform = faster_whisper.decode_audio(audio_path) # float32 audio sample
        if alignment_meta["tokenizer"] is None:
            emissions, stride = generate_emissions(
                alignment_model,
                alignment_meta,
                torch.from_numpy(audio_waveform)
                .to(torch.float32)
                .to(args.device),
                batch_size=args.batch_size,
            )
        else:
            emissions, stride = generate_emissions(
                alignment_model,
                alignment_meta,
                torch.from_numpy(audio_waveform)
                .to(alignment_model.dtype)
                .to(alignment_model.device),
                batch_size=args.batch_size,
            )

        # 用完立刻释放对齐模型，避免在长测试集上积累显存/内存碎片
        del alignment_model
        torch.cuda.empty_cache()

        tokens_starred, text_starred = preprocess_text(
            asr_output_full_transcript,
            romanize=True,
            #language=langs_to_iso[info.language],
            #language=args.language,
            language=langs_to_iso[args.language],
        )
        alignment_tokenizer = alignment_meta["tokenizer"]
        alignment_dictionary = alignment_meta["dictionary"]
        segments, scores, blank_token = get_alignments(
            emissions,
            tokens_starred,
            alignment_tokenizer,
            alignment_dictionary,
        )

        spans = get_spans(tokens_starred, segments, blank_token)

        word_timestamps = postprocess_results(text_starred, spans, stride, scores)
        return word_timestamps
    except Exception as e:
        logging.error(f"Error in add_timestampes_using_alignment_model for {audio_path}: {str(e)}")
        logging.exception("Stack trace for add_timestampes_using_alignment_model:")
        logging.error("Exiting program due to alignment model inference error.")
        sys.exit(1)

def gen_speaker_diarization_model_output(args, audio_path, models):
    try:
        speaker_ts = None
        if args.diarizer == "msdd":
            speaker_ts = gen_mssd_output(args, audio_path, models)
        elif args.diarizer == "streaming_softformer":
            speaker_ts = gen_streaming_softformer_ouput(args, audio_path, models)
        elif args.diarizer == "offline_softformer": # audio length limit,otherwise cuda mem oom
            speaker_ts = gen_offline_softformer_ouput(args, audio_path, models)
        elif args.diarizer == "pyannote31":
            speaker_ts = gen_pyannote31_output(args, audio_path, models)
        return speaker_ts
    except Exception as e:
        logging.error(f"Error in gen_speaker_diarization_model_output for {audio_path}: {str(e)}")
        logging.exception("Stack trace for gen_speaker_diarization_model_output:")
        logging.error("Exiting program due to diarization model inference error.")
        sys.exit(1)

def gen_pyannote31_output(args, audio_path, models):
    pipeline = models.get('pyannote31_pipeline')
    
    # Run diarization (pipeline is already optimized for GPU in initialize_models)
    diarization = pipeline(audio_path, min_speakers=2, max_speakers=args.max_speakers)
    
    # Convert to lab format
    lab_list = [(i) for i in diarization.to_lab().split("\n")]
    speaker_ts = []
    for i in lab_list:
        if i != "":
            start_ms = int(float(i.split()[0])*1000)
            end_ms = int(float(i.split()[1])*1000)
            speaker_num = int(i.split()[-1][-1]) # last char of "SPEAKER_00"
            speaker_ts.append((start_ms, end_ms, speaker_num))
    return speaker_ts
def gen_mssd_output(args, audio_path, models):
    audio_waveform = faster_whisper.decode_audio(audio_path) # float32 audio sample
    diarizer_model = models.get('msdd_diarizer')
    logging.info("Diarization ....")
    speaker_ts = diarizer_model.diarize(torch.from_numpy(audio_waveform).unsqueeze(0))
    return speaker_ts

def gen_streaming_softformer_ouput(args, audio_path, models):
    diar_model = models.get('streaming_softformer_model')
    logging.info("Diarization ....")
    speaker_ts = diar_model.diarize(audio=[audio_path], batch_size=1)

    labels = []
    for label in speaker_ts[0]:
        start, end, speaker = label.split() # second, second, speaker_{speaker_index}
        start, end = float(start), float(end)
        start, end = int(start * 1000), int(end * 1000)
        labels.append((start, end, int(speaker.split("_")[1])))
    labels = sorted(labels, key=lambda x: x[0])
    return labels

def gen_offline_softformer_ouput(args, audio_path, models):
    diar_model = models.get('offline_softformer_model')
    predicted_segments = diar_model.diarize(audio=audio_path, batch_size=1)
    labels = []
    for label in predicted_segments[0]:
        start, end, speaker = label.split() # second, second, speaker_{speaker_index}
        start, end = float(start), float(end)
        start, end = int(start * 1000), int(end * 1000)
        labels.append((start, end, int(speaker.split("_")[1])))
    labels = sorted(labels, key=lambda x: x[0])
    return labels


def realign_with_punc(wsm, args, models):
    try:
        #if info.language in punct_model_langs:
        if "en" in punct_model_langs:
            # restoring punctuation in the transcript to help realign the sentences
            punct_model = models.get('punct_model')

            words_list = list(map(lambda x: x["word"], wsm))

            labled_words = punct_model.predict(words_list, chunk_size=230)

            ending_puncts = ".?!"
            model_puncts = ".,;:!?"

            # We don't want to punctuate U.S.A. with a period. Right?
            is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

            for word_dict, labeled_tuple in zip(wsm, labled_words):
                word = word_dict["word"]
                if (
                    word
                    and labeled_tuple[1] in ending_puncts
                    and (word[-1] not in model_puncts or is_acronym(word))
                ):
                    word += labeled_tuple[1]
                    if word.endswith(".."):
                        word = word.rstrip(".")
                    word_dict["word"] = word

        else:
            logging.warning(
                f"Punctuation restoration is not available for the specified language."
                " Using the original punctuation."
            )
        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
        return wsm
    except Exception as e:
        logging.error(f"Error in realign_with_punc: {str(e)}")
        logging.exception("Stack trace for realign_with_punc:")
        logging.error("Exiting program due to punctuation model inference error.")
        sys.exit(1)


def initialize_models(args):
    """Initialize all models once at the beginning to avoid repeated instantiation."""
    models = {}
    
    try:
        logging.info("Initializing models...")
        
        # Initialize ASR models based on model type
        if "whisper" in args.model_dir_or_name:
            logging.info("Initializing Whisper ASR model...")
            whisper_model = faster_whisper.WhisperModel(
                args.model_dir_or_name, 
                device=args.device, 
                compute_type=mtypes[args.device],
                local_files_only=args.local_files_only,
            )
            whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
            models['whisper_model'] = whisper_model
            models['whisper_pipeline'] = whisper_pipeline
        elif "FireRedASR-LLM-L" in args.model_dir_or_name:
            logging.info("Initializing FireRedASR-LLM-L model...")
            model = FireRedAsr.from_pretrained("llm", args.model_dir_or_name)
            models['firered_llm_l_model'] = model
        elif "FireRedASR2-LLM" in args.model_dir_or_name:
            logging.info("Initializing FireRedASR2-LLM model...")
            asr_config = FireRedAsr2Config(
                use_gpu=True,
                use_half=True,
                decode_min_len=0,
                repetition_penalty=args.repetition_penalty,
                llm_length_penalty=args.llm_length_penalty,
                temperature=args.temperature,
            )
            model = FireRedAsr2.from_pretrained("llm", args.model_dir_or_name, config=asr_config)
            models['fireredasr2_model'] = model
        
        # 对齐模型很大，且在长测试集上重复使用同一个实例容易出现 CUDA OOM
        # 或内部缓存问题，这里只记录配置，在每条音频内部按需加载并释放。
        logging.info("Registering alignment model config (loaded per audio in alignment step).")
        models["alignment_config"] = {
            "device": args.device,
            "model_path": args.align_model,
            "dtype": torch.float16 if args.device == "cuda" else torch.float32,
            "local_files_only": args.ctc_model_local_files_only,
        }
        
        # Initialize diarization models based on diarizer type
        if args.diarizer == "msdd":
            logging.info("Initializing MSDD diarization model...")
            from whosaidwhat.pipe.diarization import MSDDDiarizer
            diarizer_model = MSDDDiarizer(args, device=args.device)
            models['msdd_diarizer'] = diarizer_model
        elif args.diarizer == "streaming_softformer":
            logging.info("Initializing streaming_softformer diarization model...")
            from nemo.collections.asr.models import SortformerEncLabelModel
            diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2")
            diar_model.eval()
            diar_model.sortformer_modules.chunk_len = 340
            diar_model.sortformer_modules.chunk_right_context = 40
            diar_model.sortformer_modules.fifo_len = 40
            diar_model.sortformer_modules.spkcache_update_period = 300
            models['streaming_softformer_model'] = diar_model
        elif args.diarizer == "offline_softformer":
            logging.info("Initializing offline_softformer diarization model...")
            from nemo.collections.asr.models import SortformerEncLabelModel
            diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
            diar_model.eval()
            models['offline_softformer_model'] = diar_model
        elif args.diarizer == "pyannote31":
            logging.info("Initializing pyannote-3.1 diarization model...")
            from pyannote.audio import Pipeline
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=args.hf_token
            )
            # Optimize pipeline for GPU acceleration if available
            if args.device == "cuda" and torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
            models['pyannote31_pipeline'] = pipeline
        
        # Initialize punctuation model if needed
        if args.realign_with_punc and "en" in punct_model_langs:
            logging.info("Initializing punctuation model...")
            punct_model = PunctuationModel(model=args.punct_model)
            models['punct_model'] = punct_model
        
        logging.info("All models initialized successfully.")
        return models
        
    except Exception as e:
        logging.error(f"Error initializing models: {str(e)}")
        logging.error("Exiting program due to model initialization error.")
        sys.exit(1)


def process_single_audio(args, audio_path, models, verbose=True):
    try:
        wsm, speaker_ts = gen_txt_with_ws(args, audio_path, models)
        if args.realign_with_punc:
            logging.info(f"using punc to realign....")
            wsm = realign_with_punc(wsm, args, models)

        ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

        # Prepare output filename
        audio_filename = Path(audio_path).stem
        output_path = os.path.join(args.output_folder, f"{audio_filename}.txt")
        output_path_srt = os.path.join(args.output_folder, f"{audio_filename}.srt")

        with open(f"{output_path}", "w", encoding="utf-8-sig") as f:
            get_speaker_aware_transcript(ssm, f)

        with open(f"{output_path_srt}", "w", encoding="utf-8-sig") as srt:
            write_srt(ssm, srt)

        if verbose:
            logging.info(f"Saved result to: {output_path}")
            logging.info(f"Saved result to: {output_path_srt}")
        return output_path
    except Exception as e:
        logging.error(f"Error processing audio file {audio_path}: {str(e)}")
        logging.exception("Stack trace for process_single_audio:")
        logging.error("Exiting program due to processing error.")
        sys.exit(1)
    #finally:
    #    cleanup(temp_path)

def main():
    args = parse_arguments()

    # Validate input folder
    if not os.path.exists(args.input_folder):
        logging.error(f"Error: Input folder '{args.input_folder}' does not exist.")
        sys.exit(1)

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Initialize all models once at the beginning
    models = initialize_models(args)

    # Find WAV files
    if args.use_vad:
        wav_files = find_wav_files(args.input_folder, args.file_pattern)
    else:
        wav_files = find_wav_files_from_jsonl(args.input_folder)

    total_in_dataset = len(wav_files)
    if total_in_dataset == 0:
        logging.warning("No WAV files found. Exiting.")
        return

    # Skip already decoded files (resume from previous run)
    if args.skip_existing:
        completed_stems = get_completed_stems(args.output_folder)
        wav_files = [f for f in wav_files if Path(f).stem not in completed_stems]
        skipped = total_in_dataset - len(wav_files)
        if skipped > 0:
            logging.info(
                f"Resume mode: skipping {skipped} file(s) already present in {args.output_folder}. "
                f"Remaining to process: {len(wav_files)}."
            )
        if len(wav_files) == 0:
            logging.info("All files already decoded. Nothing to do.")
            return

    # Process files
    successful_files = 0
    for audio_file in wav_files:
        result = process_single_audio(args, audio_file, models)
        if result:
            successful_files += 1

    logging.info(
        f"\nProcessing complete! Successfully processed {successful_files}/{len(wav_files)} files "
        f"(total in dataset: {total_in_dataset})."
    )
    logging.info(f"Results saved to: {args.output_folder}")

if __name__ == "__main__":
    main()
