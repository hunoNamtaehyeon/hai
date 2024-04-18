import datetime
import json
import os
from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm
import re
from fuzzywuzzy import fuzz

from make_base_dict_test import MakeBaseDictionary

from pprint import pprint


def main(base_dict, df):
    find_dict = dict()
    for user_idx, docs in tqdm(base_dict.items()):
        name = df.loc[user_idx, '지원자명']
        for doc_idx, sents in docs.items():
            for sent_idx, sent in sents.items():
                re_sent = re.sub(r'[^\w]', '', sent)
                re_name = re.sub(r'[^\w]', '', name)
                if re_name in re_sent:
                    if user_idx not in find_dict:
                        find_dict[user_idx] = {}
                        
                    if doc_idx not in find_dict[user_idx]:
                        find_dict[user_idx][doc_idx] = {}
                        
                    if sent_idx not in find_dict[user_idx][doc_idx]:
                        find_dict[user_idx][doc_idx][sent_idx] = ""
                            
                    find_dict[user_idx][doc_idx].update({sent_idx : sent.replace(name, f"!@#{name}!@#")})
    return find_dict

def make_result(df, find_dict, columns):
    chk_array = [[{} for _ in range(len(columns))] for _ in range(len(df))]
    chk_df = pd.DataFrame(chk_array, columns = [i for i in range(len(columns))] ,index=base_dict.keys())
    for user_idx, docs in tqdm(find_dict.items()):
        for doc_idx, sents in docs.items():
            chk_df.loc[user_idx, doc_idx].update(sents)
    
    chk_df.columns = [f"검출{int(i[-1])}" for i in columns]
    return chk_df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str,
                        default="../data/자기소개서_1028_20230624025957.xls")
    # parser.add_argument("--load_path2", type=str, default='none')
    parser.add_argument("--save_path", type=str, default="./outputs")
    parser.add_argument("--columns", type=str, default='자기소개서1,자기소개서2,자기소개서4')
    args = parser.parse_args()
    dict_class = MakeBaseDictionary(args)
    dict_class.preprocessing()
    base_dict = dict_class.make_base_dict()
    df = dict_class.df
    args.colname = dict_class.colname
    find_dict = main(base_dict, df)
    columns = args.colname
    resultDF = make_result(df, find_dict, columns)
    
    ##### 분석용 컬럼 -> 원래 컬럼으로 변환
    df.columns = dict_class.origin_colname
    #####
    
    finaldf = pd.concat([df, resultDF], axis=1)
    filename = '_'.join(args.load_path.split('_')[-2:]).split('.')[0]
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
    os.makedirs(args.save_path, exist_ok=True)
    finaldf.to_csv(os.path.join(args.save_path, 'blind_name' + formatted_datetime + '.csv'), 
                #    encoding='utf-8',
                   encoding='cp949',
                   errors='ignore',
                   index=None)