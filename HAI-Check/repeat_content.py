import datetime
import json
import os
from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm

from make_base_dict import MakeBaseDictionary


def cut_and_slide(sentence, min_letter):
    substrings = []
    for start in range(len(sentence) - min_letter + 1):
        substring = sentence[start:start + min_letter]
        substrings.append(substring)
    return substrings


def count_sentence_inclusion(sentence1, sentence2):
    count = 0
    start = 0

    while start < len(sentence2):
        position = sentence2.find(sentence1, start)
        if position == -1:
            break
        count += 1
        start = position + 1

    return count


def remove_duplicates_keep_order(winlist):
    relist = []
    for i in range(len(winlist)):
        if winlist[i] in winlist[i + 1:] and winlist[i] not in relist:
            relist.append(winlist[i])
    return relist


def make_win2sent(relist):
    res_sent = relist[0]
    first_let = relist[0][:3]
    for i in range(1, len(relist)):
        if res_sent[-1] == relist[i][-2]:
            res_sent += relist[i][-1]
            if res_sent[-3:] == first_let:
                return res_sent[:-3]
        else:
            break
    return res_sent


def main(df):
    newdf = pd.DataFrame(index=df.index, columns=args.colname)
    for col in args.colname:
        print('=' * 100)
        print(' ' * 50, f'{col} 진행중')
        print('=' * 100)
        for user, docs in tqdm(enumerate(df[col])):
            winlist = cut_and_slide(docs, args.min_letter)
            if len(winlist) != len(set(winlist)):
                relist = remove_duplicates_keep_order(winlist)
                dup_sent = make_win2sent(relist)
                sent_cnt = docs.count(dup_sent)
                newdf.at[user, col] = json.dumps({'count': sent_cnt, 'dup_sent': dup_sent.strip()}, ensure_ascii=False)
    newdf.columns = ['검출 ' + i[-1] for i in args.colname]

    return newdf


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str,
                        default="../data/자기소개서_1028_20230624025957.xls")
    # parser.add_argument("--load_path2", type=str, default='none')
    parser.add_argument("--save_path", type=str, default="./outputs")
    parser.add_argument("--columns", type=str, default='자기소개서1,자기소개서2,자기소개서4')
    parser.add_argument("--min_letter", type=int, default=30)
    args = parser.parse_args()

    dict_class = MakeBaseDictionary(args)
    dict_class.preprocessing()
    base_dict = dict_class.make_base_dict()
    df = dict_class.df
    args.colname = dict_class.colname

    resultDF = main(df)
    finaldf = pd.concat([df, resultDF], axis=1)
    filename = '_'.join(args.load_path.split('_')[-2:]).split('.')[0]
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
    os.makedirs(args.save_path, exist_ok=True)
    finaldf.to_csv(os.path.join(args.save_path, 'repeat_content_' + formatted_datetime + '.csv'), 
                #    encoding='utf-8',
                   encoding='cp949',
                   errors='ignore')
