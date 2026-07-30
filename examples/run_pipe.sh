#!/usr/bin/env bash
# End-to-end example: ASR + diarization + STM export + evaluation.
#
# Usage:
#   export MODEL_DIR=/path/to/FireRedASR-LLM-L
#   export ALIGN_MODEL=/path/to/mms-300m-forced-aligner
#   export INPUT_AUDIO_DIR=/path/to/audio_root
#   export OUTPUT_ROOT=/path/to/output
#   bash examples/run_pipe.sh --stage 300 --stop-stage 302
#
# Or install first: pip install -e ".[nemo,eval]"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

stage=0
stop_stage=1000
. "${REPO_ROOT}/examples/parse_options.sh"

# --- configurable paths (override via environment) ---
INPUT_AUDIO_DIR="${INPUT_AUDIO_DIR:-/path/to/complete_audio}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/pipe}"
MODEL_DIR="${MODEL_DIR:-/path/to/FireRedASR-LLM-L}"
ALIGN_MODEL="${ALIGN_MODEL:-MahmoudAshraf/mms-300m-1130-forced-aligner}"
REF_AUDIO_DIR="${REF_AUDIO_DIR:-${INPUT_AUDIO_DIR}}"
SUBSET="${SUBSET:-mlc}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

DIARIZE="python3 -m whosaidwhat.pipe.diarize_with_asr_improved"
DIARIZE2="python3 -m whosaidwhat.pipe.diarize_with_asr_improved2"
GEN_STM="python3 -m whosaidwhat.pipe.generate_hyp_stm_from_whisper_nemo_dia_output_ami_candor_fisher"
GEN_STM_ALIMEETING="python3 -m whosaidwhat.pipe.generate_hyp_stm_from_whisper_nemo_dia_output_for_alimeeting"

# Stage 300: ASR + diarization + alignment -> .txt / .srt
if [ "${stage}" -le 300 ] && [ "${stop_stage}" -ge 300 ]; then
  for line in ${SUBSET}; do
    output_dir="${OUTPUT_ROOT}/${line}"
    mkdir -p "${output_dir}"
    ${DIARIZE} \
      --model-dir-or-name "${MODEL_DIR}" \
      --local-files-only true \
      --use-vad true \
      --realign-with-punc false \
      --diarizer streaming_softformer \
      --align-model "${ALIGN_MODEL}" \
      --language en \
      --input-folder "${INPUT_AUDIO_DIR}/${line}/wav" \
      --output-folder "${output_dir}"
  done
fi

# Stage 301: collect SRT file list
if [ "${stage}" -le 301 ] && [ "${stop_stage}" -ge 301 ]; then
  echo "generate srt file list"
  for line in ${SUBSET}; do
    dest_dir="${OUTPUT_ROOT}/${line}"
    find "${dest_dir}" -iname "*.srt" > "${dest_dir}/srt_list.txt"
  done
fi

# Stage 302: SRT -> hyp.stm
if [ "${stage}" -le 302 ] && [ "${stop_stage}" -ge 302 ]; then
  for line in ${SUBSET}; do
    dest_dir="${OUTPUT_ROOT}/${line}"
    ${GEN_STM} "${dest_dir}/srt_list.txt" "${dest_dir}/hyp.stm"
  done
fi

# Stage 303: evaluation (requires meeteval: pip install meeteval)
if [ "${stage}" -le 303 ] && [ "${stop_stage}" -ge 303 ]; then
  echo "compute tcpwer, cpwer and der"
  for line in ${SUBSET}; do
    dest_dir="${OUTPUT_ROOT}/${line}"
    if [ "${line}" = ami ]; then
      cat "${REF_AUDIO_DIR}/${line}/stm/"*.stm \
        | awk 'BEGIN{OFS=" "} { $1 = $1 ".Mix-Headset"; print }' > "${dest_dir}/raw_ref.stm"
    else
      cat "${REF_AUDIO_DIR}/${line}/stm/"*.stm > "${dest_dir}/raw_ref.stm"
    fi
    meeteval-wer tcpwer -r "${dest_dir}/raw_ref.stm" -h "${dest_dir}/hyp.stm" --collar 5
    meeteval-wer cpwer -r "${dest_dir}/raw_ref.stm" -h "${dest_dir}/hyp.stm"
    meeteval-der dscore -r "${dest_dir}/raw_ref.stm" -h "${dest_dir}/hyp.stm" --collar 0.25
    meeteval-der md_eval_22 -r "${dest_dir}/raw_ref.stm" -h "${dest_dir}/hyp.stm" --collar 0.25
    meeteval-der dscore -r "${dest_dir}/raw_ref.stm" -h "${dest_dir}/hyp.stm" --collar 0.0
    meeteval-der md_eval_22 -r "${dest_dir}/raw_ref.stm" -h "${dest_dir}/hyp.stm" --collar 0.0
  done
fi

#grep -v WARNING logs/run_whisper_nemo_dia_aistation_stage303_newest.log
#nohup: ignoring input
#compute tcpwer, cpwer and der
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/fisher/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/fisher/hyp_tcpwer.json
#INFO %tcpWER: 22.11% [ 56280 / 254555, 10086 ins, 26862 del, 19332 sub ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/fisher/hyp_cpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/fisher/hyp_cpwer.json
#INFO %cpWER: 21.01% [ 53489 / 254555, 8635 ins, 25411 del, 19443 sub ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/fisher/hyp_dscore_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/fisher/hyp_dscore.json
#INFO %DER: 12.87% [ 8270.50s / 64249.57s, 6669.78s missed, 1167.63s falarm, 433.09s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/fisher/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/fisher/hyp_md_eval_22.json
#INFO %DER: 12.85% [ 8258.17s / 64249.57s, 6669.78s missed, 1155.30s falarm, 433.09s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/fisher/hyp_dscore_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/fisher/hyp_dscore.json
#INFO %DER: 23.41% [ 19893.91s / 84969.57s, 15838.72s missed, 2918.97s falarm, 1136.23s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/fisher/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/fisher/hyp_md_eval_22.json
#INFO %DER: 23.39% [ 19873.88s / 84969.57s, 15838.72s missed, 2898.94s falarm, 1136.23s spk error ]
#line: ami...
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/ami/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/ami/hyp_tcpwer.json
#INFO %tcpWER: 38.18% [ 34225 / 89635, 4960 ins, 18591 del, 10674 sub ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/ami/hyp_cpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/ami/hyp_cpwer.json
#INFO %cpWER: 31.45% [ 28188 / 89635, 2988 ins, 16619 del, 8581 sub ]
#         md-eval-22 removes the first suffix for uem filenames/session_ids but not in rttm files
#             (e.g., uem: some.audio.wav -> some.wav).
#             (e.g., rttm: some.audio.wav -> some.audio.wav).
#         dcores doesn't support dots in uem
#             (e.g., without uem file, rttm filenames/session_ids can have dots).
#             (e.g., with uem file, rttm cannot have dots).
#          -> dcores has no proper support of dots, because they use md-eval-22
#         In meeteval, we assume, that the uem file has the same filenames/session_ids as reference and hypothesis.
#          -> remove dots from filename and restore them later
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/ami/hyp_dscore_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/ami/hyp_dscore.json
#INFO %DER: 30.09% [ 7621.48s / 25325.95s, 4240.46s missed, 2760.94s falarm, 620.09s spk error ]
#         md-eval-22 removes the first suffix for uem filenames/session_ids but not in rttm files
#             (e.g., uem: some.audio.wav -> some.wav).
#             (e.g., rttm: some.audio.wav -> some.audio.wav).
#         dcores doesn't support dots in uem
#             (e.g., without uem file, rttm filenames/session_ids can have dots).
#             (e.g., with uem file, rttm cannot have dots).
#          -> dcores has no proper support of dots, because they use md-eval-22
#         In meeteval, we assume, that the uem file has the same filenames/session_ids as reference and hypothesis.
#          -> remove dots from filename and restore them later
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/ami/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/ami/hyp_md_eval_22.json
#INFO %DER: 28.42% [ 7198.89s / 25325.95s, 4240.46s missed, 2338.35s falarm, 620.09s spk error ]
#         md-eval-22 removes the first suffix for uem filenames/session_ids but not in rttm files
#             (e.g., uem: some.audio.wav -> some.wav).
#             (e.g., rttm: some.audio.wav -> some.audio.wav).
#         dcores doesn't support dots in uem
#             (e.g., without uem file, rttm filenames/session_ids can have dots).
#             (e.g., with uem file, rttm cannot have dots).
#          -> dcores has no proper support of dots, because they use md-eval-22
#         In meeteval, we assume, that the uem file has the same filenames/session_ids as reference and hypothesis.
#          -> remove dots from filename and restore them later
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/ami/hyp_dscore_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/ami/hyp_dscore.json
#INFO %DER: 37.63% [ 12227.93s / 32494.56s, 7892.37s missed, 3396.76s falarm, 938.80s spk error ]
#         md-eval-22 removes the first suffix for uem filenames/session_ids but not in rttm files
#             (e.g., uem: some.audio.wav -> some.wav).
#             (e.g., rttm: some.audio.wav -> some.audio.wav).
#         dcores doesn't support dots in uem
#             (e.g., without uem file, rttm filenames/session_ids can have dots).
#             (e.g., with uem file, rttm cannot have dots).
#          -> dcores has no proper support of dots, because they use md-eval-22
#         In meeteval, we assume, that the uem file has the same filenames/session_ids as reference and hypothesis.
#          -> remove dots from filename and restore them later
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/ami/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/ami/hyp_md_eval_22.json
#INFO %DER: 36.32% [ 11802.21s / 32494.56s, 7892.37s missed, 2971.04s falarm, 938.80s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/candor/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/candor/hyp_tcpwer.json
#INFO %tcpWER: 28.79% [ 15096 / 52426, 4750 ins, 7069 del, 3277 sub ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/candor/hyp_cpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/candor/hyp_cpwer.json
#INFO %cpWER: 27.47% [ 14399 / 52426, 4275 ins, 6594 del, 3530 sub ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/candor/hyp_dscore_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/candor/hyp_dscore.json
#INFO %DER: 19.32% [ 2608.36s / 13499.68s, 1162.45s missed, 773.39s falarm, 672.52s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/candor/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/candor/hyp_md_eval_22.json
#INFO %DER: 14.29% [ 1928.76s / 13499.68s, 1162.45s missed, 93.79s falarm, 672.52s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/candor/hyp_dscore_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/candor/hyp_dscore.json
#INFO %DER: 26.71% [ 4719.04s / 17668.34s, 2889.26s missed, 961.02s falarm, 868.76s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/candor/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/candor/hyp_md_eval_22.json
#INFO %DER: 22.85% [ 4037.86s / 17668.34s, 2889.26s missed, 279.84s falarm, 868.76s spk error ]
#compute tcpwer, cpwer and der
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/mlc/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/mlc/hyp_tcpwer.json
#INFO %tcpWER: 27.35% [ 31651 / 115745, 13355 ins, 9649 del, 8647 sub ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/mlc/hyp_cpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/mlc/hyp_cpwer.json
#INFO %cpWER: 23.18% [ 26831 / 115745, 11299 ins, 7593 del, 7939 sub ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/mlc/hyp_dscore_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/mlc/hyp_dscore.json
#INFO %DER: 15.02% [ 5264.72s / 35048.32s, 422.94s missed, 2665.04s falarm, 2176.73s spk error ]
#WARNING No UEM file provided. See https://github.com/fgnt/meeteval/issues/97#issuecomment-2508140402 for details.
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/mlc/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/mlc/hyp_md_eval_22.json
#INFO %DER: 13.35% [ 4679.81s / 35048.32s, 422.94s missed, 2080.13s falarm, 2176.73s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/mlc/hyp_dscore_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/mlc/hyp_dscore.json
#INFO %DER: 21.38% [ 8503.05s / 39762.83s, 1368.81s missed, 4500.74s falarm, 2633.50s spk error ]
#WARNING No UEM file provided. See https://github.com/fgnt/meeteval/issues/97#issuecomment-2508140402 for details.
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/mlc/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/firered-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_english/mlc/hyp_md_eval_22.json
#INFO %DER: 19.89% [ 7907.21s / 39762.83s, 1368.81s missed, 3904.90s falarm, 2633.50s spk error ]



if [ ${stage} -le 330 ] && [ ${stop_stage} -ge 330 ];then
   input_audio_dir=/F00120240032/alimeeting/
   subset="Eval_Ali/Eval_Ali_far/audio_dir Test_Ali/Test_Ali_far/audio_dir"
   #subset="Eval_Ali/Eval_Ali_far/audio_dir"
   export HF_ENDPOINT=https://hf-mirror.com
   for name in $subset;do
      subfolder=`echo "$name" | awk -F/ '{print $2}'`
      output_dir=/maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/$subfolder
      ${DIARIZE2} \
              --model-dir-or-name /maduo/model_hub/FireRedASR2-LLM  \
              --fireredvad-dir /maduo/model_hub/FireRedVAD/VAD\
              --repetition-penalty 1.0\
              --llm-length-penalty 0.0\
              --temperature 1.0\
              --local-files-only true\
              --use-vad true\
              --realign-with-punc false\
              --diarizer streaming_softformer\
              --align-model /maduo/model_hub/MahmoudAshraf/mms-300m-1130-forced-aligner\
              --language zh \
              --input-folder  $input_audio_dir/$name\
              --output-folder $output_dir\
              --skip-existing false
   done
fi

if [ ${stage} -le 331 ] && [ ${stop_stage} -ge 331 ];then
  echo "generate srt files"
  subset="Eval_Ali/Eval_Ali_far/audio_dir Test_Ali/Test_Ali_far/audio_dir"
  #subset="Test_Ali/Test_Ali_far/audio_dir"
  for line in $subset;do
      subfolder=`echo "$line" | awk -F/ '{print $2}'`
      dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/$subfolder
      find $dest_dir -iname "*.srt" > $dest_dir/srt_list.txt
  done
fi
if [ ${stage} -le 332 ] && [ ${stop_stage} -ge 332 ];then
   subset="Eval_Ali/Eval_Ali_far/audio_dir Test_Ali/Test_Ali_far/audio_dir"
   #subset="Test_Ali/Test_Ali_far/audio_dir"
   input_audio_dir=/F00120240032/alimeeting/
   for line in $subset;do
     subfolder=`echo "$line" | awk -F/ '{print $2}'`
     dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/$subfolder
     #input=$dest_dir/srt_list.txt
     #output=$dest_dir/hyp.stm
     input=$dest_dir/srt_list.txt
     output=$dest_dir/hyp.stm
     output_jsonl=$dest_dir/hyp.jsonl
     audio_dir=$input_audio_dir/$line
     ${GEN_STM_ALIMEETING} \
             $input $audio_dir $output $output_jsonl
   done
fi


if [ ${stage} -le 333 ] && [ ${stop_stage} -ge 333 ];then
   echo "compute tcpwer, cpwer and der"
   input_dir=/F00120240032/alimeeting
   subset="Eval_Ali/Eval_Ali_far Test_Ali/Test_Ali_far"
   #subset="Test_Ali/Test_Ali_far"
   for line in $subset;do
     subfolder=`echo "$line" | awk -F/ '{print $2}'`
     dest_dir=/maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/$subfolder
     meeteval-wer tcpwer -r  $input_dir/$line/stm/alimeeting_${subfolder}.stm -h $dest_dir/hyp.stm --collar 5
     meeteval-wer cpwer -r  $input_dir/$line/stm/alimeeting_${subfolder}.stm -h $dest_dir/hyp.stm
     meeteval-der dscore -r  $input_dir/$line/stm/alimeeting_${subfolder}.stm -h $dest_dir/hyp.stm  --collar 0.25
     meeteval-der md_eval_22 -r  $input_dir/$line/stm/alimeeting_${subfolder}.stm -h $dest_dir/hyp.stm  --collar 0.25
     meeteval-der dscore -r  $input_dir/$line/stm/alimeeting_${subfolder}.stm -h $dest_dir/hyp.stm  --collar 0.0
     meeteval-der md_eval_22 -r  $input_dir/$line/stm/alimeeting_${subfolder}.stm -h $dest_dir/hyp.stm  --collar 0.0
     done
fi

#bash run_whisper_nemo_dia_aistation.sh --stage 333 --stop-stage 333
#compute tcpwer, cpwer and der
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Eval_Ali_far/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Eval_Ali_far/hyp_tcpwer.json
#INFO %tcpWER: 44.45% [ 36008 / 81005, 2745 ins, 22916 del, 10347 sub ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Eval_Ali_far/hyp_cpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Eval_Ali_far/hyp_cpwer.json
#INFO %cpWER: 40.91% [ 33143 / 81005, 1583 ins, 21754 del, 9806 sub ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Eval_Ali_far/hyp_dscore_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Eval_Ali_far/hyp_dscore.json
#INFO %DER: 19.84% [ 2243.06s / 11307.25s, 1839.69s missed, 317.77s falarm, 85.60s spk error ]
#WARNING No UEM file provided. See https://github.com/fgnt/meeteval/issues/97#issuecomment-2508140402 for details.
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Eval_Ali_far/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Eval_Ali_far/hyp_md_eval_22.json
#INFO %DER: 19.54% [ 2209.18s / 11307.25s, 1839.69s missed, 283.89s falarm, 85.60s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Eval_Ali_far/hyp_dscore_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Eval_Ali_far/hyp_dscore.json
#INFO %DER: 32.44% [ 5720.83s / 17632.60s, 4578.43s missed, 825.41s falarm, 316.99s spk error ]
#WARNING No UEM file provided. See https://github.com/fgnt/meeteval/issues/97#issuecomment-2508140402 for details.
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Eval_Ali_far/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Eval_Ali_far/hyp_md_eval_22.json
#INFO %DER: 32.24% [ 5684.64s / 17632.60s, 4578.43s missed, 789.22s falarm, 316.99s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Test_Ali_far/hyp_tcpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Test_Ali_far/hyp_tcpwer.json
#INFO %tcpWER: 43.90% [ 91790 / 209072, 5670 ins, 58739 del, 27381 sub ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Test_Ali_far/hyp_cpwer_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Test_Ali_far/hyp_cpwer.json
#INFO %cpWER: 40.89% [ 85489 / 209072, 3317 ins, 56386 del, 25786 sub ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Test_Ali_far/hyp_dscore_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Test_Ali_far/hyp_dscore.json
#INFO %DER: 20.15% [ 5868.06s / 29126.12s, 4633.21s missed, 934.97s falarm, 299.88s spk error ]
#WARNING No UEM file provided. See https://github.com/fgnt/meeteval/issues/97#issuecomment-2508140402 for details.
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Test_Ali_far/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Test_Ali_far/hyp_md_eval_22.json
#INFO %DER: 19.82% [ 5771.62s / 29126.12s, 4633.21s missed, 838.53s falarm, 299.88s spk error ]
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Test_Ali_far/hyp_dscore_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Test_Ali_far/hyp_dscore.json
#INFO %DER: 32.89% [ 14749.56s / 44840.78s, 11625.12s missed, 2208.54s falarm, 915.90s spk error ]
#WARNING No UEM file provided. See https://github.com/fgnt/meeteval/issues/97#issuecomment-2508140402 for details.
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Test_Ali_far/hyp_md_eval_22_per_reco.json
#INFO Wrote: /maduo/exp/asr_sd/whisper_nemo_dia/fireredasr2-llm_mms_300m_ctc_aligner_with_streaming_softformer_diarization_infer_for_alimeeting/Test_Ali_far/hyp_md_eval_22.json
#INFO %DER: 32.66% [ 14646.30s / 44840.78s, 11625.12s missed, 2105.27s falarm, 915.90s spk error ]

