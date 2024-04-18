import datetime
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


def is_hate_speech(tokenizer, model, sent):
    inputs = tokenizer(sent, return_tensors='pt', truncation=True)
    input_ids = inputs['input_ids'].to(args.device)
    attention_mask = inputs['attention_mask'].to(args.device)
    outputs = model(input_ids,
                    attention_mask=attention_mask)
    logits = outputs['logits']
    results = torch.softmax(logits, dim=-1)[0][0]
    # 이 값이 0.5보다 크면 일반문장, 그렇지 않으면 비속어 및 혐오문장임.
    if results >= 0.5:
        return False, round(float(results), 5)
    else:

        return True, round(float(results), 5)


def make_output_dict(tokenizer, model, base_dict):
    output_dict = {}
    for u, rows in tqdm(base_dict.items()):
        for col, docs in rows.items():
            for si, sent in docs.items():
                for mali in malicious:
                    if mali in sent:
                        if u not in output_dict:
                            output_dict[u] = {}
                        if f"검출{dict_class.colname[col][-1]}" not in output_dict[u]:
                            output_dict[u][f"검출{dict_class.colname[col][-1]}"] = {}
                        if si + 1 not in output_dict[u][f"검출{dict_class.colname[col][-1]}"]:
                            output_dict[u][f"검출{dict_class.colname[col][-1]}"][si + 1] = {}
                        output_dict[u][f"검출{dict_class.colname[col][-1]}"][si + 1] = json.dumps({
                            'score': is_hate_speech(tokenizer, model, sent)[1],
                            'keyword': mali,
                            'DB_filter': True,
                            'AI_filter': is_hate_speech(tokenizer, model, sent)[0],
                            'sent': sent}, ensure_ascii=False)
    return output_dict


def infer(args):
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.device)
    MODEL_NAME = "klue/roberta-large"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 2
    model.config = model_config
    model.classifier.out_proj.out_features = 2
    model.load_state_dict(torch.load(args.best_ckpt))
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    output_dict = make_output_dict(tokenizer, model, base_dict)
    output_df = pd.DataFrame.from_dict(output_dict, orient='index').reindex(df.index).fillna(0)
    return output_df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str, default="../data/자기소개서_1111_20230630030038.xls")
    parser.add_argument("--save_path", type=str, default="./outputs")
    parser.add_argument("--best_ckpt", type=str, default="./models/bisok2/best.pt")
    parser.add_argument("--screen_path", type=str, default="./add_DB/mali_screen.xlsx")
    parser.add_argument("--columns", type=str, default='자기소개서1,자기소개서4')
    args = parser.parse_args()

    dict_class = MakeBaseDictionary(args)
    # dict_class.preprocessing()
    base_dict = dict_class.make_base_dict()
    df = dict_class.df

    malicious = pd.read_excel(args.screen_path, engine='openpyxl')['비속어'].dropna()
    malicious = list(set(malicious))
    output_df = infer(args)
    final_df = pd.concat([df, output_df], axis=1)
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
    os.makedirs(args.save_path, exist_ok=True)
    final_df.to_csv(os.path.join(args.save_path, 'bisok_' + formatted_datetime + '.csv'),
                    encoding='utf-8',
                    errors='ignore',
                    index=None)
