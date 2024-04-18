import datetime
import json
import os
import re
from argparse import ArgumentParser

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from transformers import AutoTokenizer

from make_base_dict import MakeBaseDictionary
from simcse_model import SimCSE


def simcse_emb(sent, simcse_model, tokenizer):
    sent_emb = tokenizer(sent,
                         return_tensors="pt",
                         truncation=True,
                         add_special_tokens=True,
                         max_length=128).to(args.device)

    with torch.no_grad():
        simcse_embedding = simcse_model.encode(sent_emb, args.device)
    return simcse_embedding


def keep_only_korean_words(text):
    korean_pattern = re.compile('[가-힣]+')
    korean_words = korean_pattern.findall(text)
    cleaned_text = ' '.join(korean_words)
    return cleaned_text


def make_simcse_emb(df):
    model_name = 'klue/roberta-large'
    simcse_model = SimCSE(args, mode='test').to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tqdm.pandas()
    emb_colname = []
    for col_num in range(len(dict_class.colname)):
        emb_colname.append(f'prepro_simcse_{col_num}')
        df[f'prepro_simcse_{col_num}'] = df[dict_class.colname[col_num]].progress_apply(lambda x: np.mean(([
            np.array(torch.flatten(simcse_emb(keep_only_korean_words(sent), simcse_model, tokenizer)).cpu()) for sent in
            dict_class.tokenize_sentences(x)]), axis=0))
    return df, emb_colname


def cal_loss_from_emb(new_embedded_doc):
    cosim_lst = []
    emb_mean = torch.mean(new_embedded_doc, axis=0)
    total_mean = emb_mean * len(new_embedded_doc)
    for i in tqdm(range(len(new_embedded_doc))):
        targets = new_embedded_doc[i:i + 1]
        new_mean = ((total_mean - targets) / (len(new_embedded_doc) - 1)).reshape((1, -1))
        cosim_lst.append(1 - cosine_similarity(targets, new_mean).item())
    return cosim_lst


def make_score_col(df, simtest):
    output_lst = []
    for roconcat in simtest:
        output_lst.append(cal_loss_from_emb(roconcat))
    for col_num in range(len(dict_class.colname)):
        df[f'스코어{col_num + 1}'] = output_lst[col_num]
    return df


def make_newscore_col(df):
    scaler = MinMaxScaler()
    pvals = []
    for col_num in range(len(dict_class.colname)):
        score_noscale = 100 - (df[[f'스코어{col_num + 1}']] * 100)
        df[f'뉴스코어{col_num + 1}'] = score_noscale
    return df


def check_with_threshold(score, threshold):
    if score <= threshold:
        return True
    else:
        return False


def newscore2result(x, mean, min, Q1, Q2, Q3):
    if x < 0:
        return json.dumps(
            {'score': 0, 'result': check_with_threshold(x, 25), 'mean': mean, 'min': min, 'Q1': Q1, 'Q2': Q2,
             'Q3': Q3}, ensure_ascii=False)
    else:
        return json.dumps(
            {'score': round(x, 2), 'result': check_with_threshold(x, 25), 'mean': mean, 'min': min, 'Q1': Q1,
             'Q2': Q2, 'Q3': Q3}, ensure_ascii=False)


def make_dfscore(df):
    add_col = []
    for col_num in range(len(dict_class.colname)):
        add_col.append(f'검출{dict_class.colname[col_num][-1]}')
        mean = round(df[f'뉴스코어{col_num + 1}'].mean(), 2)
        min = round(df[f'뉴스코어{col_num + 1}'].min(), 2)
        Q1 = round(df[f'뉴스코어{col_num + 1}'].quantile(.25), 2)
        Q2 = round(df[f'뉴스코어{col_num + 1}'].quantile(.5), 2)
        Q3 = round(df[f'뉴스코어{col_num + 1}'].quantile(.75), 2)
        df[f'검출{dict_class.colname[col_num][-1]}'] = df[f'뉴스코어{col_num + 1}'].apply(
            lambda x: newscore2result(x, mean, min, Q1, Q2, Q3))
    return df, add_col


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str, default="../data/자기소개서_1111_20230630030038.xls")
    parser.add_argument("--save_path", type=str, default="./outputs")
    parser.add_argument("--weight_path", type=str, default="./models")
    parser.add_argument("--test_model_name", type=str, default='nomatter')
    parser.add_argument("--columns", type=str, default='자기소개서1,자기소개서4')
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dict_class = MakeBaseDictionary(args)
    dict_class.preprocessing()
    base_dict = dict_class.make_base_dict()
    df = dict_class.df
    default_colname = df.columns
    df, emb_colname = make_simcse_emb(df)
    simtest = [torch.stack([torch.tensor(row) for row in df[col].values]) for col in emb_colname]
    df = make_score_col(df, simtest)
    df = make_newscore_col(df)
    df, add_col = make_dfscore(df)
    final_col = list(default_colname) + add_col
    findf = df[final_col]
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
    os.makedirs(args.save_path, exist_ok=True)
    findf.to_csv(os.path.join(args.save_path, 'nomatter_' + formatted_datetime + '.csv'),
                 encoding='utf-8',
                 errors='ignore',
                 index=None)
