# from konlpy.tag import Okt
# import os
# import shutil

# okt = Okt()

# print(okt.pos("순대국 먹고 싶다."))
# print(okt.pos("순댓국 먹고 싶다."))
# print(okt.pos("패스트파이브에서 일을 합니다."))
# print(okt.pos("아이오아이는 정말 이뻐요."))

# directory = '/mnt/c/Users/USER/Desktop/nam/hai/HAI-Check/venv/lib/python3.10/site-packages/konlpy/java'
# os.chdir(directory)
# os.getcwd() 

# # !jar xvf open-korean-text-2.1.0.jar
# import subprocess

# # jar 파일 추출
# subprocess.run(['jar', 'xvf', 'open-korean-text-2.1.0.jar'], check=True)
# # data 확인
# with open(os.path.join(directory, "org/openkoreantext/processor/util/noun/names.txt")) as f:
#     data = f.read()
    
# print(data)

# data += '순댓국\n순대국\n아이오아이\n패스트파이브\n'

# # 사전 저장
# with open(os.path.join(directory, "org/openkoreantext/processor/util/noun/names.txt"), 'w') as f:
#     f.write(data)
    
# with open(os.path.join(directory, "org/openkoreantext/processor/util/noun/names.txt")) as f:
#     data = f.read()
# print(data)

# # !jar cvf open-korean-text-2.1.0.jar org
# # !rm -r org
# subprocess.run(['jar', 'cvf', 'open-korean-text-2.1.0.jar', 'org'], cwd=directory, check=True)
# shutil.rmtree(f'{directory}/org')

from test0 import add_dict
from konlpy.tag import Okt
add_dict()
okt = Okt()

print(okt.pos("기업명 먹고 싶다."))
print(okt.pos("한전KPS 먹고 싶다."))
print(okt.pos("패스트파이브에서 일을 합니다."))
print(okt.pos("아이오아이는 정말 이뻐요."))















from itertools import combinations, permutations
import copy
detected_dict_copy = copy.deepcopy(detected_dict)
detected_dict_test = copy.deepcopy(detected_dict)
tokenizer = Okt()
window_pass_pos = ['Punctuation', 'Josa', 'Verb', 'Adjective', 'Noun']
window_pass_pos = list(permutations(window_pass_pos,1)) + \
                    list(permutations(window_pass_pos,2))
for k, v in detected_dict_copy.items():
    print(k, v)
    print()
    space = True if " " in k else False
    for v_v in v:
        if k in v_v:  
            print(v_v)
            detected_com = v_v.replace(k, "기업명")
            chk_com = ('기업명', 'Noun')
            pos_tag_list = tokenizer.pos(detected_com)
            com_idx = pos_tag_list.index(chk_com)
            start_idx, end_idx = (0,2) if com_idx == 0 else (com_idx-1, com_idx+2)
            window_pos_tag_list = pos_tag_list[start_idx: end_idx]
            print(pos_tag_list)
            print(window_pos_tag_list)
            window_pos_tag_list = [i for i in window_pos_tag_list if i != chk_com]
            print(window_pos_tag_list)
            if len(window_pos_tag_list) == 0:
                pop_idx = detected_dict_test[k].index(v_v)
                detected_dict_test[k].pop(pop_idx)
            else:
                chk_window_pos = tuple([i[-1] for i in window_pos_tag_list])
                if chk_window_pos in window_pass_pos:
                    pop_idx = detected_dict_test[k].index(v_v)
                    detected_dict_test[k].pop(pop_idx)
                    print("잘못 없음!")
            
        #     pos_list = [i[0] for i in tokenizer.pos(v_v)]
        #     if space:
        #         pos_list_len = [len("".join(pos_list[:idx+1])) for idx, i in enumerate(pos_list)]
        #         for pdx, p_l in enumerate(pos_list_len):
        #             if p_l == v_v.index(" "):
        #                 pos_list.insert(pdx+1, " ")
        #     print("***", pos_list)
        #     print("***", tokenizer.pos(v_v))
        #     for n in range(1, len(pos_list)+1):
        #         if k in ["".join(i) for i in list(combinations(pos_list, n))]:
        #             print(k)
        #             pop_index = detected_dict[k].index(v_v)
        #             detected_dict[k].pop(pop_index)

            print("-"*50)   
    print()