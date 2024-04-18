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


def is_gender(tokenizer, model, sent, args):
    inputs = tokenizer(sent, return_tensors='pt', truncation=True).to(args.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs['logits']
    results = torch.softmax(logits, dim=-1)
    # print(results)
    return int(torch.argmax(results)), float(max(results[0]))


def no_genders_sent(sent, gender_lists):
    found = [word for word in gender_lists if word in sent]
    if found:
        return True, found[0]
    return False, None


def make_infer_sent_dict(tokenizer, model, base_dict, gender_keys, gender_ban, gender_ban_keys, args):
    infer_sent_dict = {'sent': [], 'index': [], 'gender_name': [], 'confidence': []}
    for u, rows in tqdm(base_dict.items()):
        for col, docs in rows.items():
            for si, sent in docs.items():
                corp_check = no_genders_sent(sent, gender_keys)
                if corp_check[0]:
                    if corp_check[1] in gender_ban_keys:  # 단어가 문장에 포함되어있으면 ban키에 포함 되었는지 확인
                        ban_check = no_genders_sent(sent, gender_ban[corp_check[1]])
                        if not ban_check[0]:
                            pred = is_gender(tokenizer, model, sent, args)
                            if pred[0] == 0 and pred[1] > 0.9:
                                infer_sent_dict['sent'].append(sent)
                                infer_sent_dict['index'].append((u, col, si + 1))
                                infer_sent_dict['gender_name'].append(corp_check[1])
                                infer_sent_dict['confidence'].append(pred[1])
                    elif corp_check[1] not in gender_ban_keys:
                        pred = is_gender(tokenizer, model, sent, args)
                        if pred[0] == 0 and pred[1] > 0.9:
                            infer_sent_dict['sent'].append(sent)
                            infer_sent_dict['index'].append((u, col, si + 1))
                            infer_sent_dict['gender_name'].append(corp_check[1])
                            infer_sent_dict['confidence'].append(pred[1])
    return infer_sent_dict


def make_output_dict(infer_sent_dict, args):
    output_dict = {}
    for si in tqdm(range(len(infer_sent_dict['sent']))):
        sent = infer_sent_dict['sent'][si]
        index = infer_sent_dict['index'][si]
        gender_name = infer_sent_dict['gender_name'][si]
        confi = infer_sent_dict['confidence'][si]
        ru = index[0]
        rcol = index[1]
        rsi = index[2]
        if ru not in output_dict:
            output_dict[ru] = {}
        if f"검출{rcol}" not in output_dict[ru]:
            output_dict[ru][f"검출{rcol}"] = {}
        if rsi not in output_dict[ru][f"검출{rcol}"]:
            output_dict[ru][f"검출{rcol}"][rsi] = {}
        output_dict[ru][f"검출{rcol}"][rsi] = json.dumps({'confidence': round(confi * 100, 2),
                                                        'gender_name': gender_name,
                                                        'class': args.label_decode[0],
                                                        'sent': sent}, ensure_ascii=False)

    return output_dict


def infer(base_dict, gender_keys, gender_ban, gender_ban_keys, args):
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
    infer_sent_dict = make_infer_sent_dict(tokenizer, model, base_dict, gender_keys, gender_ban, gender_ban_keys, args)
    output_dict = make_output_dict(infer_sent_dict, args)
    return output_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str, default="../data/killer.xlsx")
    parser.add_argument("--save_path", type=str, default="./outputs")
    parser.add_argument("--best_ckpt", type=str, default="./models/blind_gender_best.pt")
    parser.add_argument("--gender_db", type=str, default="./add_DB/gender_db.xlsx")
    parser.add_argument("--gender_ban", type=str, default="./add_DB/gender_ban.xlsx")
    parser.add_argument("--columns", type=str, default='full')
    args = parser.parse_args()
    args.label_decode = {0: '위반', 1: '위반아님', 2: '다른단어'}

    dict_class = MakeBaseDictionary(args)
    dict_class.preprocessing()
    base_dict = dict_class.make_base_dict()
    df = dict_class.df

    gender_name_df = pd.read_excel(args.gender_db, engine='openpyxl')[['스크린 키워드']]
    gender_name_df = gender_name_df.rename(columns={'스크린 키워드': 'gender_key'})
    gender_keys = gender_name_df['gender_key'].dropna().tolist()
    gender_keys = sorted(gender_keys, key=len, reverse=True)
    banpair = pd.read_excel(args.gender_ban, engine='openpyxl')
    banpair = banpair.rename(columns={'금지어': 'gender_ban'})
    gender_ban = banpair.groupby('gender_key')['gender_ban'].apply(list).to_dict()
    gender_ban_keys = gender_ban.keys()
    left_tot_keys = list(gender_keys - gender_ban_keys)
    gender_ban_keys = list(gender_ban_keys)

    output_dict = infer(base_dict, gender_keys, gender_ban, gender_ban_keys, args)

    outputdf_col = [f'검출{int(i[-1])}' for i in dict_class.colname if i.startswith('자기소개서')]
    output_df = pd.DataFrame.from_dict(output_dict, columns=outputdf_col, orient='index').reindex(df.index).fillna(0)
    final_df = pd.concat([df, output_df], axis=1)

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
    os.makedirs(args.save_path, exist_ok=True)
    final_df.to_csv(os.path.join(args.save_path, 'blind_gender_' + formatted_datetime + '.csv'),
                    encoding='utf-8',
                    errors='ignore',
                    index=None)
