import re
from argparse import ArgumentParser

import nltk
import pandas as pd

nltk.download('punkt')  # Download the tokenizer data


class MakeBaseDictionary:
    def __init__(self, args):
        ##### self.df, self.df1_len = self.df_loader(args.load_path)
        ##### 원본 컬럼명까지 불러오기
        self.df, self.df1_len, self.origin_colname, self.origin_user_ind = self.df_loader(args.load_path)
        #####

        self.colname_whole = [col for col in self.df.columns if col.startswith('자기소개서')]
        if args.columns == 'full':
            self.colname = self.colname_whole
        else:
            self.input_col = [item.strip() for item in args.columns.split(',')]
            is_included = all(item in self.colname_whole for item in self.input_col)
            self.colname = self.input_col
            
            ##### 자기소개서 순서 정렬
            self.colname = sorted(self.colname)
            #####
            
            if not is_included:
                raise Exception("데이터에 존재하는 컬럼명만을 입력해주세요.")
        # df 리딩 - 전체 컬럼 파싱 - 인풋 컬럼 파싱 -
        if 'load_path2' in args and args.load_path2.lower() != 'none':
            # if args.load_path2.lower()!='none': #second path 존재한다면?
            
            ##### self.df2, self.df2_len = self.df_loader(args.load_path2)
            ##### DB2 불러올 때 원래 컬럼명은 안 씀.
            self.df2, self.df2_len, _ = self.df_loader(args.load_path2)
            #####
            
            self.df = pd.concat([self.df, self.df2], axis=0).reset_index(drop=True)
            
        ##### print(len(self.df), self.colname)수
        ##### 분석 대상 출력
        # print()
        # print(" * 분석 대상 로우 개수 :", len(self.df))
        # print()
        q_index = [int(i[-1]) for i in self.colname]
        q_target = [self.origin_colname[j+2] for i,j in zip(range(len(self.origin_colname)), q_index)]
        # for c, q_t in zip(self.colname, q_target):
        #     print(" * 분석 대상 컬럼 : " + c + " <=> " + q_t)
        # print()
        # print(" * 원본 전체 컬럼")
        # print(self.origin_colname.values)
        #####

    def preprocessing(self):
        print('start preprocessing')
        for col in self.colname_whole:
            self.df[col] = self.df[col].apply(lambda row: self.prepro_ser(row))

    def df_loader(self, load_path):
        formats = load_path.split('.')[-1]
        if formats == 'xlsx':
            loaded_df = pd.read_excel(load_path, engine='openpyxl').fillna(' ').astype(str)
            
            ##### return loaded_df, len(loaded_df)
            ##### 원본 컬럼 저장 / 원본 컬럼 -> 분석용 컬럼으로 변환
            origin_colname = loaded_df.columns
            loaded_df.columns = ['수험번호', '지원분야', '지원자명'] + [ "자기소개서"+ str(i+1) for i in range(len(loaded_df.columns)-3)]
            origin_user_ind = loaded_df['수험번호']
            return loaded_df, len(loaded_df), origin_colname, origin_user_ind
            #####
            
        elif formats == 'xls':
            loaded_df = pd.read_excel(load_path).fillna(' ').astype(str)
            
            ##### return loaded_df, len(loaded_df)
            ##### 원본 컬럼 저장 / 원본 컬럼 -> 분석용 컬럼으로 변환
            origin_colname = loaded_df.columns
            loaded_df.columns = ['수험번호', '지원분야', '지원자명'] + [ "자기소개서"+ str(i+1) for i in range(len(loaded_df.columns)-3)]
            origin_user_ind = loaded_df['수험번호']
            return loaded_df, len(loaded_df), origin_colname, origin_user_ind
            #####
            
            
    def prepro_ser(self, row):
        row = row.rstrip()
        if len(row) != 0:
            if row[-1] != '.':
                row = row + '.'
        else:
            row = row + '.'
        return row

    def tokenize_sentences(self, document):
        try:
            # sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', document) # 숫자. 의 형식을 문장으로 간주
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<!\d\.)(?<=\.|\?)\s', document) # 숫자. 의 형식을 문장으로 간주하지 않음.
        except:
            sentences = ['nan']
        return sentences

    def make_base_dict(self):
        user_num = len(self.df)
        capture_dict = {}
        for user in range(user_num):
            capture_dict[user] = {}
            for col_num, col in enumerate(self.colname):  # 인풋컬럼 넣어야함
                capture_dict[user][col_num] = {}
                document = self.df.loc[user, self.colname[col_num]]
                sents = self.tokenize_sentences(document)
                for sent_num, sent in enumerate(sents):
                    capture_dict[user][col_num][sent_num] = sent
        return capture_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str, default="../data/killer.xlsx")
    parser.add_argument("--load_path2", type=str, default="../data/killer.xlsx")
    parser.add_argument("--save_path", type=str, default="./")
    # parser.add_argument("--columns", type=str, default="자기소개서3,자기소개서4")
    parser.add_argument("--columns", type=str, default="full")
    args = parser.parse_args()

    dict_class = MakeBaseDictionary(args)
    dict_class.preprocessing()
    base_dict = dict_class.make_base_dict()
    df = dict_class.df
    df1_len = dict_class.df1_len
