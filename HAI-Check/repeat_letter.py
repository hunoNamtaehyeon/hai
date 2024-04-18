import datetime
import json
import os
import re
from argparse import ArgumentParser

import nltk
import pandas as pd
from soynlp.tokenizer import RegexTokenizer
from tqdm import tqdm

from make_base_dict import MakeBaseDictionary

nltk.download('punkt')  # Download the tokenizer data


def has_repeated_more_than_four(word):
    word = word.lower()
    pattern = r'(\W)\1{4,}|(\w)\2{4,}'  # Matches four or more repeated letters
    return bool(re.search(pattern, word))


def new_check_letter(sent, tokenizer):
    sent = re.sub(r"\bwww\.\S+", "", sent)
    splist = tokenizer.tokenize(sent)
    splist.append(splist[-1])  # 계산 편리성을 위해 뒤에 마지막 원소 하나 더 추가
    results_lsts = []
    matches = []
    next_lsts = []
    for i in range(len(splist) - 1):
        word = splist[i]
        next_word = splist[i + 1]
        if has_repeated_more_than_four(word):
            results_lsts.append(True)
            matches.append(word)
            next_lsts.append(next_word)
        else:
            results_lsts.append(False)
    return any(results_lsts), matches, next_lsts


def detect_repeated_words_blank(sentence):
    words = sentence.split()
    for i in range(len(words) - 2):
        if words[i] == words[i + 1] == words[i + 2] and len(words[i]) >= 2:
            return True, words[i]

    return False, ""


def detect_repeated_blnk_words(sentence):
    words = sentence.split()
    words_leng = len(words)
    words_set_leng = len(set(words))
    key = ' '.join(list(set(words)))
    if words_set_leng / words_leng < 0.5:
        return True, key
    return False, ""


def detect_repeated_words_dot(sentence):
    words = sentence.split('.')
    for i in range(len(words) - 2):
        if words[i] == words[i + 1] == words[i + 2] and len(words[i]) >= 2:
            return True, words[i]

    return False, ""


def detect_repeated_words_nospace(sentence):
    match = re.search(r'(\w{2,})\1{2,}', sentence)  # {2,}{2,}면 두글자 이상 3번이상 반복을 의미.

    if match:
        return True, match.group(1)
    else:
        return False, ""


def pairs_bool_check(*pairs):
    result = []
    for pair in pairs:
        if pair[0]:
            result.append(pair[1])
            break
    return pair[0], result


def repeat_word_dict(base_dict):
    sum_dict = {}

    for user_ind, row in base_dict.items():
        for docs_ind, docu in row.items():
            for sent_ind, sent in docu.items():
                result_bool_space_blank = detect_repeated_words_blank(sent)
                result_bool_space_dot = detect_repeated_words_dot(sent)
                result_bool_blnk = detect_repeated_blnk_words(sent)
                result_bool_nospace = detect_repeated_words_nospace(sent)
                result_bool = pairs_bool_check(result_bool_space_blank, result_bool_space_dot, result_bool_nospace,
                                               result_bool_blnk)

                if result_bool[0]:
                    if len(result_bool[1][0]) / len(sent) >= 0.5 or len(
                            result_bool[1][0]) >= 10:  # or len(set(result_bool[1][0])):
                        if user_ind not in sum_dict:
                            sum_dict[user_ind] = {}
                        if docs_ind not in sum_dict[user_ind]:
                            sum_dict[user_ind][docs_ind] = {}
                        if sent_ind not in sum_dict[user_ind][docs_ind]:
                            sum_dict[user_ind][docs_ind][sent_ind] = {'key': result_bool[1][0], 'string': sent}
                        else:
                            sum_dict[user_ind][docs_ind][sent_ind] = {'key': result_bool[1][0], 'string': sent}
    return sum_dict


def repeat_letter_counting(base_dict):
    tokenizer = RegexTokenizer()
    sum_dict = {}
    for user_ind, row in tqdm(base_dict.items()):
        for docs_ind, docu in row.items():
            for sent_ind, sent in docu.items():
                result_pair = new_check_letter(sent, tokenizer)
                # print(result_pair, sent, ex)
                if result_pair[0]:
                    if user_ind not in sum_dict:
                        sum_dict[user_ind] = {}
                    if docs_ind not in sum_dict[user_ind]:
                        sum_dict[user_ind][docs_ind] = {}
                        # print(sum_dict)
                    if sent_ind not in sum_dict[user_ind][docs_ind]:
                        sum_dict[user_ind][docs_ind][sent_ind] = []
                        sum_dict[user_ind][docs_ind][sent_ind].append(result_pair[1][0])
                        sum_dict[user_ind][docs_ind][sent_ind].append(result_pair[2][0])
                    else:
                        sum_dict[user_ind][docs_ind][sent_ind].append(result_pair[1][0])
                        sum_dict[user_ind][docs_ind][sent_ind].append(result_pair[2][0])
    return sum_dict


def repeat_letter_dict(sum_dict):
    non_filt = []
    letter_repeat_dict = {}
    for u, d in sum_dict.items():
        for doc, v in d.items():
            for s, l in v.items():
                string = base_dict[u][doc][s]
                sum_dict_key = sum_dict[u][doc][s].pop(0)
                string_len = len(string)
                key_len = len(sum_dict_key)
                if key_len / string_len >= 0.5:
                    if u not in letter_repeat_dict:
                        letter_repeat_dict[u] = {}
                    if doc not in letter_repeat_dict[u]:
                        letter_repeat_dict[u][doc] = {}
                    if s not in letter_repeat_dict[u][doc]:
                        letter_repeat_dict[u][doc][s] = {'key': sum_dict_key, 'string': string}
                    else:
                        letter_repeat_dict[u][doc][s] = {'key': sum_dict_key, 'string': string}
    return letter_repeat_dict


def union_dicts(dict1, dict2):
    dict3 = dict1.copy()

    for key in dict2:
        if key in dict3:
            if isinstance(dict3[key], dict) or isinstance(dict2[key], dict):
                # Recursively union nested dictionaries
                dict3[key] = union_dicts(dict3[key], dict2[key])
        else:
            dict3[key] = dict2[key]

    return dict3


def dict_to_output(df, union_repeat_dict):
    pyo_sent = {}
    for i in range(len(args.colname)):
        pyo_sent[f'검출{i + 1}'] = [0] * len(df)
    for user_ind in union_repeat_dict.keys():
        for doc_ind, value in union_repeat_dict[user_ind].items():
            for sent_ind, sent in value.items():
                pyo_sent[f'검출{doc_ind + 1}'][user_ind] = (sent_ind, json.dumps(sent, ensure_ascii=False))
    adf = pd.DataFrame(pyo_sent)
    outputDF = pd.concat([df, adf], axis=1)
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
    os.makedirs(args.save_path, exist_ok=True)
    outputDF.to_csv(os.path.join(args.save_path, 'repeat_letter_' + formatted_datetime + '.csv'),
                    encoding='utf-8',
                    index=None,
                    errors='ignore')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str, default="../data/killer.xlsx")
    parser.add_argument("--save_path", type=str, default="./outputs")
    # parser.add_argument("--columns", type=str, default='자기소개서1,자기소개서2,자기소개서4')
    parser.add_argument("--columns", type=str, default='full')
    args = parser.parse_args()

    dict_class = MakeBaseDictionary(args)
    dict_class.preprocessing()
    base_dict = dict_class.make_base_dict()
    args.colname = dict_class.colname
    print(len(base_dict.keys()), 'basedic 길이')
    # 단어 반복 검출
    word_repeat_dict = repeat_word_dict(base_dict)
    # 글자 반복 검출
    sum_dict = repeat_letter_counting(base_dict)
    letter_repeat_dict = repeat_letter_dict(sum_dict)
    # 단어, 글자 반복 통합
    union_repeat_dict = union_dicts(letter_repeat_dict, word_repeat_dict)
    # with open('./union_repeat_dict.json', 'w') as outfile:
    #     json.dump(union_repeat_dict, outfile)
    dict_to_output(dict_class.df, union_repeat_dict)
    # print(union_repeat_dict)
