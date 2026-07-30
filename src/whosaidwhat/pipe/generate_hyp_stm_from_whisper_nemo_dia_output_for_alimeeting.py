import sys
import re
import argparse
from pathlib import Path
import pysrt
import json
import re
import unicodedata
from whisper_normalizer.english import EnglishTextNormalizer

# 文本归一化配置
CASE_SENSITIVE = False
TO_CHAR = True  # 英文False，中文True
IGNORE_WORDS = set()
REMOVE_TAG = True
SPACELIST = {' ', '\t', '\r', '\n'}
ALL_PUNCTS = set([
    '!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/',
    ':', ';', '=', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '}', '~',
    '、', '。', '！', '，', '；', '？', '：', '「', '」', '︰', '『', '』', '《', '》'
])

def stripoff_tags(text):
    """移除所有<xxx>格式的标签"""
    if not text:
        return ''
    return re.sub(r'<[^>]+>', '', text)

def remove_all_puncts(text):
    """移除所有标点符号"""
    if not text:
        return ""
    return ''.join([c for c in text if c not in ALL_PUNCTS])

def characterize(string):
    """字符级归一化（中文用）"""
    res = []
    i = 0
    while i < len(string):
        char = string[i]
        if char in ALL_PUNCTS:
            i += 1
            continue
        cat1 = unicodedata.category(char)
        if cat1 == 'Zs' or cat1 == 'Cn' or char in SPACELIST:
            i += 1
            continue
        if cat1 == 'Lo':
            res.append(char)
            i += 1
        else:
            sep = ' '
            if char == '<':
                sep = '>'
            j = i + 1
            while j < len(string):
                c = string[j]
                if ord(c) >= 128 or (c in SPACELIST) or (c == sep):
                    break
                j += 1
            if j < len(string) and string[j] == '>':
                j += 1
            res.append(string[i:j])
            i = j
    return res

def normalize_text(text):
    """核心文本归一化函数（修复：避免重复归一化/冗余空格）"""
    if not text:
        return ""

    # 1. 移除标签
    text = stripoff_tags(text)
    if not text:
        return ""

    # 2. 英文专用归一化（whisper_normalizer）
    english_normalizer = EnglishTextNormalizer()
    text = english_normalizer(text)

    # 3. 移除标点
    text = remove_all_puncts(text)
    if not text:
        return ""

    # 4. 大小写转换
    if not CASE_SENSITIVE:
        text = text.upper()

    # 5. 分词/分字符
    if TO_CHAR:
        tokens = characterize(text)
    else:
        # 修复：合并多空格为单空格，避免空token
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split() if text else []

    # 6. 过滤忽略词+空token
    ignore_words_upper = {w.upper() for w in IGNORE_WORDS} if not CASE_SENSITIVE else IGNORE_WORDS
    normalized_tokens = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token in ignore_words_upper:
            continue
        normalized_tokens.append(token)

    # 修复：避免末尾多余空格
    return ' '.join(normalized_tokens)


def convert_to_second(inp:str):
    """
    "00:00:01,432" -> 1.4320
    """
    a = inp
    tot = int(a.split(":")[0])*3600 + int(a.split(":")[1])*60 + int(a.split(":")[2].split(',')[0])+ int(a.split(":")[2].split(',')[1])/1000
    return round(tot,4)
# 时间字符串转毫秒的工具函数
def time_str_to_ms(time_str):
    """将00:00:00,400格式的时间转换为毫秒"""
    h, m, s = time_str.split(':')
    sec, ms = s.split(',')
    total_ms = int(h) * 3600000 + int(m) * 60000 + int(sec) * 1000 + int(ms)
    return total_ms

def text_normalization(input_line_text):
    custom_punctuations = r'!"#$%&()*+,./:;<=>?@[\\]^_`{|}~。、？！・¿¡，'
    punctuation_pattern = re.compile(f'[{re.escape(custom_punctuations)}]')
    input_line_text = input_line_text.strip()
    ori_text = input_line_text.lower()
    text_tn = punctuation_pattern.sub('', ori_text)
    text_tn = re.sub(r' +', ' ', text_tn)
    return text_tn

def normalize_text_alimeeting(text: str, normalize: str = "m2met") -> str:
    """
    Text normalization similar to M2MeT challenge baseline.
    See: https://github.com/yufan-aslp/AliMeeting/blob/main/asr/local/text_normalize.pl
    """
    if normalize == "none":
        return text
    elif normalize == "m2met":
        import re

        text = text.replace("<sil>", "")
        text = text.replace("<%>", "")
        text = text.replace("<->", "")
        text = text.replace("<$>", "")
        text = text.replace("<#>", "")
        text = text.replace("<_>", "")
        text = text.replace("<space>", "")
        text = text.replace("`", "")
        text = text.replace("&", "")
        text = text.replace(",", "")
        if re.search("[a-zA-Z]", text):
            text = text.upper()
        text = text.replace("Ａ", "A")
        text = text.replace("ａ", "A")
        text = text.replace("ｂ", "B")
        text = text.replace("ｃ", "C")
        text = text.replace("ｋ", "K")
        text = text.replace("ｔ", "T")
        text = text.replace("，", "")
        text = text.replace("丶", "")
        text = text.replace("。", "")
        text = text.replace("、", "")
        text = text.replace("？", "")
        return text


def gen_hyp_stm(inp_srt_file_list: str, output: str):
    with open(inp_srt_file_list, "r", encoding='utf-8') as fin, open(output, 'w', encoding='utf-8')as fw:
        for line in fin:
            line = line.strip()
            uttid = Path(line).stem
            fsrt = pysrt.open(line)
            for lin in fsrt:
                if ":" in str(lin.text):
                    # normal case:
                    # 1
                    # 00:00:00,940 --> 00:00:06,360
                    # Speaker 0: What do you think about shopping these days on the internet?
                    spkid = str(lin.text).split(':')[0]
                    #spkid = re.sub(r"\[",'',spkid)
                    #spkid =  re.sub(r"\]",'',spkid)
                    spkid = "".join(spkid.split())
                    text = normalize_text_alimeeting(str(lin.text).split(':')[-1])
                    text = normalize_text(text)
                    start_s = convert_to_second(str(lin.start))
                    end_s = convert_to_second(str(lin.end))
                    if start_s>end_s:
                        tmp = start_s
                        start_s = end_s
                        end_s = tmp
                    fw.write(f"{uttid} 1 {spkid} {start_s} {end_s} {text}\n")
                else:
                    # for case:
                    #231
                    #00:18:51,958 --> 00:18:52,178
                    #Yeah, yeah.
                    spkid = "SPEAKER_empty"
                    text = normalize_text_alimeeting(str(lin.text))
                    text = normalize_text(text)
                    start_s = convert_to_second(str(lin.start))
                    end_s = convert_to_second(str(lin.end))
                    if start_s>end_s:
                        tmp = start_s
                        start_s = end_s
                        end_s = tmp
                    fw.write(f"{uttid} 1 {spkid} {start_s} {end_s} {text}\n")

def convert_srt_files_to_single_jsonl(srt_list_file, output_jsonl_path, audio_dir=None):
    """
    将指定目录下的所有SRT文件转换并写入同一个JSONL文件（每行对应一个SRT文件）
    
    参数:
        srt_list_file (str/Path): 包含所有SRT文件的文本
        output_jsonl_path (str/Path): 最终输出的单个JSONL文件路径
        audio_dir (str/Path, 可选): 音频文件所在目录，默认为SRT文件同目录
    """
    # 处理路径参数
    #srt_dir = Path(srt_dir)
    output_jsonl_path = Path(output_jsonl_path)
    audio_dir = Path(audio_dir) 
    
    
    # 创建输出目录（如果不存在）
    output_jsonl_path.parent.mkdir(exist_ok=True, parents=True)
    
    # 正则表达式匹配SRT条目
    srt_pattern = re.compile(
        r'(\d+)\n'
        r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n'
        r'Speaker (\d+): (.+?)\n\n',
        re.DOTALL
    )
    # 打开输出的JSONL文件，逐行写入每个SRT文件的转换结果
    with open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
        # 遍历目录下所有.srt文件
        #srt_files = list(srt_dir.glob("*.srt"))
        srt_files = [i.strip() for i in open(srt_list_file).readlines()]
        
        processed_count = 0
        for srt_file in srt_files:
            try:
                # 1. 构建音频文件路径（SRT文件名 = 音频文件名）
                audio_filename = Path(srt_file).stem
                audio_path = str(audio_dir / audio_filename)
                
                # 2. 读取并解析SRT文件
                with open(srt_file, 'r', encoding='utf-8') as f:
                    srt_content = f.read()
                
                matches = srt_pattern.findall(srt_content)
                if not matches:
                    print(f"警告：文件 {srt_file} 中未匹配到任何字幕条目，跳过")
                    continue
                
                # 3. 合并当前SRT文件的文本和时间戳
                combined_txt = ""
                combined_timestamp = []
                for match in matches:
                    _, start_time_str, end_time_str, spk_id, text = match
                    # 转换时间戳
                    start_s = convert_to_second(start_time_str)
                    end_s = convert_to_second(end_time_str)
                    
                    start_ms = time_str_to_ms(start_time_str)
                    end_ms = time_str_to_ms(end_time_str)
                    combined_timestamp.append([start_s, end_s])
                    # 拼接文本
                    combined_txt += f"<|{start_ms}|> {normalize_text_alimeeting(text.strip())} <|{end_ms}|><|spk{spk_id}|> " 
                # 4. 构建JSON条目并写入文件
                json_entry = {
                    "wav": audio_path,
                    "txt": combined_txt,
                    "timestamps": combined_timestamp
                }
                # 写入一行（ensure_ascii=False 保证中文正常显示）
                jsonl_file.write(json.dumps(json_entry, ensure_ascii=False) + '\n')
                
                processed_count += 1
            except Exception as e:
                print(f"处理文件 {srt_file} 时出错: {e}，跳过该文件")
        
        print(f"转换完成！共处理 {processed_count} 个SRT文件，输出到：{output_jsonl_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert AliMeeting SRT outputs to STM/JSONL for evaluation."
    )
    parser.add_argument("input", help="Text file listing SRT paths (one per line)")
    parser.add_argument("audio_dir", help="Directory containing reference audio files")
    parser.add_argument("output", help="Output STM file path")
    parser.add_argument("output_jsonl", help="Output JSONL file path")
    args = parser.parse_args()
    gen_hyp_stm(args.input, args.output)
    convert_srt_files_to_single_jsonl(args.input, args.output_jsonl, audio_dir=args.audio_dir)


if __name__ == "__main__":
    main()

