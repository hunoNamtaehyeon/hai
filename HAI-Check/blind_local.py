import os
import datetime
import json
import os
from argparse import ArgumentParser

import pandas as pd
import torch
from torch.optim.lr_scheduler import *
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, \
    AutoModelForSequenceClassification

from make_base_dict import MakeBaseDictionary


def is_local(tokenizer, model, sent, args):
    inputs = tokenizer(sent,
                       return_tensors='pt',
                       padding=True,
                       truncation=True).to(args.device)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs['logits']
    results = torch.softmax(logits, dim=-1)
    return int(torch.argmax(results)), float(max(results[0]))


def no_locals_sent(sent, local_lists):
    found = [word for word in local_lists if word in sent]
    if found:
        return True, found[0]
    return False, None


def make_infer_sent_dict(base_dict, local_lists):
    infer_sent_dict = {'sent': [], 'index': [], 'local_name': []}
    for u, rows in tqdm(base_dict.items()):
        for col, docs in rows.items():
            for si, sent in docs.items():
                check_local = no_locals_sent(sent, local_lists)
                if check_local[0]:
                    infer_sent_dict['sent'].append(sent)
                    infer_sent_dict['index'].append((u, col, si + 1))
                    infer_sent_dict['local_name'].append(check_local[1])
    return infer_sent_dict


def make_output_dict(tokenizer, model, infer_sent_dict, args):
    output_dict = {}
    for si in tqdm(range(len(infer_sent_dict['sent']))):
        sent = infer_sent_dict['sent'][si]
        index = infer_sent_dict['index'][si]
        local_name = infer_sent_dict['local_name'][si]
        pred = is_local(tokenizer, model, sent, args)
        ru = index[0]
        rcol = index[1]
        rsi = index[2]
        if pred[0] == 0 and pred[1] > 0.9:
            if ru not in output_dict:
                output_dict[ru] = {}
            if f"검출{dict_class.colname[rcol][-1]}" not in output_dict[ru]:
                output_dict[ru][f"검출{dict_class.colname[rcol][-1]}"] = {}
            if rsi not in output_dict[ru][f"검출{dict_class.colname[rcol][-1]}"]:
                output_dict[ru][f"검출{dict_class.colname[rcol][-1]}"][rsi] = {}
            output_dict[ru][f"검출{dict_class.colname[rcol][-1]}"][rsi] = json.dumps(
                {'confidence': round(pred[1] * 100, 2),
                 'local_name': local_name,
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
    infer_sent_dict = make_infer_sent_dict(base_dict, args.local_name_list)
    output_dict = make_output_dict(tokenizer, model, infer_sent_dict, args)

    return output_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str, default="../data/killer.xlsx")
    parser.add_argument("--save_path", type=str, default="./outputs")
    parser.add_argument("--best_ckpt", type=str, default="./models/blind_local_best.pt")
    parser.add_argument("--local_db", type=str, default="./add_DB/local_db.xlsx")
    parser.add_argument("--columns", type=str, default='자기소개서1, 자기소개서4')
    args = parser.parse_args()
    args.label_decode = {0: '위배', 1: '위배아님', 2: '다른단어'}

    dict_class = MakeBaseDictionary(args)
    dict_class.preprocessing()
    base_dict = dict_class.make_base_dict()
    df = dict_class.df

    # local name list
    args.local_name_list = pd.read_excel(args.local_db, engine='openpyxl')['지역명'].values.tolist()

    output_dict = infer(base_dict, args)
    outputdf_col = [f'검출{int(i[-1])}' for i in dict_class.colname if i.startswith('자기소개서')]
    output_df = pd.DataFrame.from_dict(output_dict, columns=outputdf_col, orient='index').reindex(df.index).fillna(0)
    final_df = pd.concat([df, output_df], axis=1)

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
    os.makedirs(args.save_path, exist_ok=True)
    final_df.to_csv(os.path.join(args.save_path, 'blind_local_' + formatted_datetime + '.csv'),
                    encoding='utf-8',
                    errors='ignore',
                    index=None)
