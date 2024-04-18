#  python pyojeol_self.py \
#  --load_path "./data/killer.xlsx" \
#  --save_path "./outputs" \
#  --columns "자기소개서1,자기소개서4" \
#  --slide_length 6

#  python3 pyojeol_full_test.py \
#  --load_path "./data/원격로컬비교용_철도200명.xlsx" \
#  --load_path2 "none" \
#  --save_path "./outputs/memory" \
#  --columns "full" \
#  --user_ind_colname "NO" \
#  --min_counts 2 \
#  --min_word_len 2 \
#  --percentile 90\
#  --num_processes 20 \
#  --slide_length 5

# python moonhang.py \
# --load_path "./data/killer.xlsx" \
# --save_path "./outputs" \
#  # --question

#  #OK
#  python repeat_letter.py \
#  --load_path "./data/killer.xlsx" \
#  --save_path "./outputs" \
#  --columns "full"

#  #OK
#  python count_letter_sent.py \
#  --load_path "./data/killer.xlsx" \
#  --save_path "./outputs" \
#  --letter_lower 100 \
#  --letter_upper 1000 \
#  --sent_len_lower 3 \
#  --sent_len_upper 10 \
#  --columns "full"

#  #OK
#  python3 no_matter.py \
#  --load_path "./data/테스트데이터.xlsx" \
#  --save_path "./outputs" \
#  --weight_path "./models" \
#  --test_model_name "nomatter" \
#  --columns "자기소개서1,자기소개서4"

#  #OK
#  python repeat_content.py \
#  --load_path "./data/killer.xlsx" \
#  --save_path "./outputs" \
#  --columns "자기소개서1,자기소개서4" \
#  --min_letter 30

# # new
#  python repeat_content_new.py \
#  --load_path "./data/원격로컬비교용_철도200명.xlsx" \
#  --save_path "./outputs/new" \
#  --columns "full" \
#  --min_letter 10

# # new
#  python find_keyword.py \
#  --load_path "./data/철도표절원본.xlsx" \
#  --save_path "./outputs/find_keyword" \
#  --columns "full" \
#  --keywords "대 중교통서 비스, 사회적가치를 실현하기 위해" 

# new
 python blind_name.py \
 --load_path "./data/철도표절원본.xlsx" \
 --save_path "./outputs/name" \
 --columns "full" \

#  #OK
#  python3 bisok.py \
#  --load_path "./data/테스트데이터.xlsx" \
#  --save_path "./outputs" \
#  --best_ckpt "./models/bisok/best.pt" \
#  --screen_path "./add_DB/mali_screen.xlsx" \
#  --columns "자기소개서1,자기소개서4"

#  #OK
#  python jasamyeong.py \
#  --load_path "./data/자기소개서_1111_20230630030038.xls" \
#  --save_path "./outputs" \
#  --columns "자기소개서1,자기소개서4" \
#  --company_name "한국토지주택공사"

#  #OK
#  python3 apply_other_com.py \
#  --load_path "./data/테스트데이터.xlsx" \
#  --save_path "./outputs" \
#  --best_ckpt "./models/other_com/best.pt" \
#  --comDB_path "./add_DB/other_com_comdb.xlsx" \
#  --banpairDB_path "./add_DB/other_com_banlist.xlsx" \
#  --columns "full" \
#  --corps_list "조폐공사,KOMSCO,한국조폐공사"

#  #OK
#  python3 blind_gender.py \
#  --load_path "./data/테스트데이터.xlsx" \
#  --save_path "./outputs" \
#  --best_ckpt "./models/blind_gender/best.pt" \
#  --gender_db "./add_DB/gender_db.xlsx" \
#  --gender_ban "./add_DB/gender_ban.xlsx" \
#  --columns "full"

#  #OK
#  python3 blind_schl.py \
#  --load_path "./data/테스트데이터.xlsx" \
#  --save_path "./outputs" \
#  --best_ckpt "./models/blind_schl/best.pt" \
#  --schl_db "./add_DB/schl_db.xlsx" \
#  --columns "full"

#  #OK
#  python3 blind_local.py \
#  --load_path "./data/테스트데이터.xlsx" \
#  --save_path "./outputs" \
#  --best_ckpt "./models/blind_local/best.pt" \
#  --local_db "./add_DB/local_db.xlsx" \
#  --columns "full"

# # #OK
#  python3 blind_fam_job.py \
#  --load_path "./data/테스트데이터.xlsx" \
#  --save_path "./outputs" \
#  --best_ckpt "./models/blind_famjob/best.pt" \
#  --famjob_db "./add_DB/familyDB.xlsx" \
#  --famjob_ban "./add_DB/fam_job_ban.xlsx" \
#  --columns "full"

