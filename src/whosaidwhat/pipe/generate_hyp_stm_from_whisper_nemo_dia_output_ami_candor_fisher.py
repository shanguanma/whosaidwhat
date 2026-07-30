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
TO_CHAR = False  # 英文False，中文True
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

def text_normalization(input_line_text):
    custom_punctuations = r'!"#$%&()*+,./:;<=>?@[\\]^_`{|}~。、？！・¿¡，'
    punctuation_pattern = re.compile(f'[{re.escape(custom_punctuations)}]')
    input_line_text = input_line_text.strip()
    ori_text = input_line_text.lower()
    text_tn = punctuation_pattern.sub('', ori_text)
    text_tn = re.sub(r' +', ' ', text_tn)
    return text_tn

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
                    start_s = convert_to_second(str(lin.start))
                    end_s = convert_to_second(str(lin.end))
                    trans = text_normalization(str(lin.text).split(':')[-1])
                    trans = normalize_text(trans)
                    if start_s>end_s:
                        tmp = start_s
                        start_s = end_s
                        end_s = tmp
                    fw.write(f"{uttid} 1 {spkid} {start_s} {end_s} {trans}\n")
                else:
                    # for case:
                    #231
                    #00:18:51,958 --> 00:18:52,178
                    #Yeah, yeah.
                    spkid = "SPEAKER_empty"
                    start_s = convert_to_second(str(lin.start))
                    end_s = convert_to_second(str(lin.end))
                    trans = text_normalization(str(lin.text).split(':')[-1])
                    trans = normalize_text(trans)
                    if start_s>end_s:
                        tmp = start_s
                        start_s = end_s
                        end_s = tmp
                    fw.write(f"{uttid} 1 {spkid} {start_s} {end_s} {trans}\n")

                    #fw.write(f"{uttid} 1 {spkid} {convert_to_second(str(lin.start))} {convert_to_second(str(lin.end))} {text_normalization(str(lin.text))}\n")






def main():
    parser = argparse.ArgumentParser(
        description="Convert SRT outputs to STM format for evaluation."
    )
    parser.add_argument("input", help="Text file listing SRT paths (one per line)")
    parser.add_argument("output", help="Output STM file path")
    args = parser.parse_args()
    gen_hyp_stm(args.input, args.output)


if __name__ == "__main__":
    main()


