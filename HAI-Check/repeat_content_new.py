import datetime
import json
import os
from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm

from make_base_dict_test import MakeBaseDictionary

from pprint import pprint

def main(df, base_dict):
    chk_array = [[{} for _ in range(len(args.colname))] for _ in range(len(df))]
    chk_df = pd.DataFrame(chk_array, columns = [i for i in range(len(args.colname))] ,index=base_dict.keys())
    for user_idx, docs in tqdm(base_dict.items()):
        for doc_idx, sents in docs.items():
            target_doc_sents = list(sents.values())
            if len(target_doc_sents) != len(set(target_doc_sents)):
                set_sents = list(set(sents.values()))
                for sent in set_sents:
                    sent_token = len(sent.split(" "))
                    duplicated_count = target_doc_sents.count(sent)
                    if sent_token > args.min_letter and duplicated_count > 1:
                        sdx = list(sents.values()).index(sent)
                        chk_result = {sdx : {"count" : duplicated_count, "dup_sent" : sent.strip()}}
                        # chk_df.loc[user_idx, doc_idx].append(str(chk_result))
                        # chk_df.loc[user_idx, doc_idx] = list(set(chk_df.loc[user_idx, doc_idx]))
                        chk_df.loc[user_idx, doc_idx].update(chk_result)
                        # chk_df.loc[user_idx, doc_idx] = list(set(chk_df.loc[user_idx, doc_idx]))

    chk_df.columns = [f"검출{int(i[-1])}" for i in args.colname]
    
    # print(type(chk_df.iloc[0,0]), chk_df.iloc[0,0])
    # print(type(chk_df.iloc[-1,-1]), chk_df.iloc[-1,-1])
    # chk_df[chk_df == {}] = 0
    
    return chk_df



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
    dict_class.colname
    resultDF = main(df, base_dict)
    
    ##### 분석용 컬럼 -> 원래 컬럼으로 변환
    df.columns = dict_class.origin_colname
    #####
    
    finaldf = pd.concat([df, resultDF], axis=1)
    filename = '_'.join(args.load_path.split('_')[-2:]).split('.')[0]
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
    os.makedirs(args.save_path, exist_ok=True)
    finaldf.to_csv(os.path.join(args.save_path, 'repeat_content_' + formatted_datetime + '.csv'), 
                #    encoding='utf-8',
                   encoding='cp949',
                   errors='ignore')