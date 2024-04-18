import datetime
import json
import os
import re
from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm

from make_base_dict import MakeBaseDictionary


def counting_letter(document):
    leng = len(document)
    return leng


def tokenize_sentences(document):
    try:
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', document)
    except:
        sentences = ['nan']
    return sentences


def make_counting_dict(df):
    print('Making count_dict....')
    user_num = len(df)
    count_dict = {}
    for user in range(user_num):
        count_dict[user] = {}
        for col_num, col in enumerate(args.colname):
            count_dict[user][col_num] = {'letter_count': 0, 'sent_count': 0}
            document = df.loc[user, args.colname[col_num]]
            doc_leng = counting_letter(document)
            sent_leng = len(tokenize_sentences(document))
            count_dict[user][col_num]['letter_count'] += doc_leng
            count_dict[user][col_num]['sent_count'] += sent_leng

    return count_dict


def check_count_dict2(count_dict, condition):  # 전체 다 있고, 앞에 True, False 있는 버전
    output_dict = {}
    for user, rows in tqdm(count_dict.items()):

        for col, val in rows.items():
            mini_dict = {}
            if user not in output_dict:
                output_dict[user] = {}
            if col not in output_dict[user]:
                output_dict[user][col] = mini_dict
            letter_detect = val['letter_count'] < condition['letter_lower'] or \
                            val['letter_count'] > condition['letter_upper']
            sent_detect = val['sent_count'] < condition['sent_len_lower'] or \
                          val['sent_count'] > condition['sent_len_upper']
            if letter_detect == True and sent_detect == False:
                output_dict[user][col]['글자검출'] = True
                output_dict[user][col]['문장검출'] = False
                output_dict[user][col]['글자수'] = int(val['letter_count'])
                output_dict[user][col]['문장수'] = int(val['sent_count'])
            elif letter_detect == False and sent_detect == True:
                output_dict[user][col]['글자검출'] = False
                output_dict[user][col]['문장검출'] = True
                output_dict[user][col]['글자수'] = int(val['letter_count'])
                output_dict[user][col]['문장수'] = int(val['sent_count'])
            elif letter_detect == True and sent_detect == True:
                output_dict[user][col]['글자검출'] = True
                output_dict[user][col]['문장검출'] = True
                output_dict[user][col]['글자수'] = int(val['letter_count'])
                output_dict[user][col]['문장수'] = int(val['sent_count'])
            else:
                output_dict[user][col]['글자검출'] = False
                output_dict[user][col]['문장검출'] = False
                output_dict[user][col]['글자수'] = int(val['letter_count'])
                output_dict[user][col]['문장수'] = int(val['sent_count'])

    return output_dict


def dict_transform(output_dict):
    transformed_data = {}
    for key, inner_dict in output_dict.items():
        inner_data = {}
        for inner_key, values_dict in inner_dict.items():
            for value_keys in values_dict.keys():
                inner_data[f'Q{inner_key + 1}_{value_keys}'] = values_dict[value_keys]
        transformed_data[key] = inner_data
    countdf = pd.DataFrame.from_dict(json.dumps(transformed_data, ensure_ascii=False), orient='index')
    return countdf


def origin_dict_transform(output_dict):
    transformed_data = {}
    for key, inner_dict in output_dict.items():
        inner_data = {}
        for inner_key, values_dict in inner_dict.items():
            inner_data[f'검출{args.colname[inner_key][-1]}'] = json.dumps(values_dict, ensure_ascii=False)

        transformed_data[key] = inner_data
    countdf = pd.DataFrame.from_dict(transformed_data, orient='index')
    return countdf


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str,
                        default="../data/killer.xlsx")
    parser.add_argument("--save_path", type=str, default="./outputs")
    parser.add_argument("--columns", type=str, default='자기소개서1,자기소개서3,자기소개서4')
    parser.add_argument("--letter_lower", type=int, default=100)
    parser.add_argument("--letter_upper", type=int, default=1000)
    parser.add_argument("--sent_len_lower", type=int, default=3)
    parser.add_argument("--sent_len_upper", type=int, default=10)
    args = parser.parse_args()
    condition = {'letter_lower': args.letter_lower,
                 'letter_upper': args.letter_upper,
                 'sent_len_lower': args.sent_len_lower,
                 'sent_len_upper': args.sent_len_upper}

    dict_class = MakeBaseDictionary(args)
    df = dict_class.df
    args.colname = dict_class.colname

    count_dict = make_counting_dict(df)
    output_dict = check_count_dict2(count_dict, condition)
    countdf = origin_dict_transform(output_dict)
    finaldf = pd.concat([df, countdf], axis=1)
    filename = '_'.join(args.load_path.split('_')[-2:]).split('.')[0]
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
    os.makedirs(args.save_path, exist_ok=True)
    finaldf.to_csv(os.path.join(args.save_path, 'letter_sent_count_' + formatted_datetime + '.csv'), index=None,
                   encoding='utf-8',
                   errors='ignore')
