import os
import datetime
import json
import os
from argparse import ArgumentParser

import pandas as pd
import torch
from torch.optim.lr_scheduler import *
# huno/notebook/screening/data/screen.xlsx
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, \
    AutoModelForSequenceClassification

from make_base_dict import MakeBaseDictionary


def is_schl(tokenizer, model, sent, args):
    inputs = tokenizer(sent, return_tensors='pt', truncation=True).to(args.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs['logits']
    results = torch.softmax(logits, dim=-1)
    # print(results)
    return int(torch.argmax(results)), float(max(results[0]))


def no_schls_sent(sent, schl_lists):
    for schl in schl_lists:
        if schl in sent:
            return True, schl
    return False, None


#
# def make_infer_sent_dict(base_dict, schl_lists):
#     infer_sent_dict = {'sent':[], 'index':[], 'schl_name':[]}
#     for u, rows in tqdm(base_dict.items()):
#         for col, docs in rows.items():
#             for si, sent in docs.items():
#                 check_schl = no_schls_sent(sent, schl_lists)
#                 if check_schl[0]:
#                     infer_sent_dict['sent'].append(sent)
#                     infer_sent_dict['index'].append((u, col, si+1))
#                     infer_sent_dict['schl_name'].append(check_schl[1])
#     return infer_sent_dict

def make_output_dict(tokenizer, model, base_dict, args):
    output_dict = {}
    for u, rows in tqdm(base_dict.items()):
        for col, docs in rows.items():
            for si, sent in docs.items():
                check_schl = no_schls_sent(sent, args.schl_name_list)
                if check_schl[0]:  # true란 얘기 : 학교명이 들어간 문장이라는 뜻
                    pred = is_schl(tokenizer, model, sent, args)
                    if pred[0] == 0 and pred[1] > 0.9:
                        if u not in output_dict:
                            output_dict[u] = {}
                        if f"검출{col + 1}" not in output_dict[u]:
                            output_dict[u][f"검출{col + 1}"] = {}
                        if si + 1 not in output_dict[u][f"검출{col + 1}"]:
                            output_dict[u][f"검출{col + 1}"][si + 1] = {}
                        output_dict[u][f"검출{col + 1}"][si + 1] = json.dumps({'confidence': round(pred[1] * 100, 2),
                                                                             'shl_name': check_schl[1],
                                                                             'class': args.label_decode[pred[0]],
                                                                             'sent': sent}, ensure_ascii=False)

    return output_dict


def infer(base_dict, args):
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.device)

    MODEL_NAME = "klue/roberta-large"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    model.classifier.out_proj.out_features = 3
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 3
    model.config = model_config

    model.load_state_dict(torch.load(args.best_ckpt))
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    output_dict = make_output_dict(tokenizer, model, base_dict, args)
    return output_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str, default="../data/killer.xlsx")
    parser.add_argument("--save_path", type=str, default="./outputs")
    parser.add_argument("--best_ckpt", type=str, default="./models/blind_schl/best.pt")
    parser.add_argument("--schl_db", type=str, default="./add_DB/schl_db.xlsx")
    parser.add_argument("--columns", type=str, default='full')
    args = parser.parse_args()
    args.label_decode = {0: '블라인드 위반', 1: '타학교 단순 언급', 2: '그 외'}

    dict_class = MakeBaseDictionary(args)
    dict_class.preprocessing()
    base_dict = dict_class.make_base_dict()
    df = dict_class.df

    # schl name list
    args.schl_name_list = pd.read_excel(args.schl_db, engine='openpyxl')['대학명'].values.tolist()

    output_dict = infer(base_dict, args)
    outputdf_col = [f'검출{int(i[-1])}' for i in dict_class.colname if i.startswith('자기소개서')]
    output_df = pd.DataFrame.from_dict(output_dict, columns=outputdf_col, orient='index').reindex(df.index).fillna(0)
    final_df = pd.concat([df, output_df], axis=1)

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
    os.makedirs(args.save_path, exist_ok=True)
    final_df.to_csv(os.path.join(args.save_path, 'blind_schl_' + formatted_datetime + '.csv'),
                    encoding='utf-8',
                    errors='ignore',
                    index=None)
