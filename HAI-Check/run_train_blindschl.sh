# 비속어
# python3 train_main.py \
# --data_path "./train_data/bisok_train.csv" \
# --text_col "text" \
# --label_col "label" \
# --config_file_path './config.yml' \
# --config "bisok"

# # 타기업 입사지원
# python3 train_main.py \
# --data_path "./train_data/other_com_train.csv" \
# --text_col "text" \
# --label_col "new_label" \
# --config_file_path './config.yml' \
# --config "other_com"

# # # 블라인드-가족직업
# python3 train_main.py \
# --data_path "./train_data/blind_famjob.xlsx" \
# --text_col "text" \
# --label_col "라벨" \
# --config_file_path './config.yml' \
# --config "blind_famjob"

# # 블라인드-성별
# python train_main.py \
# --data_path "./train_data/blind_gender.xlsx" \
# --text_col "text" \
# --label_col "라벨" \
# --config_file_path './config.yml' \
# --config "blind_gender"

# # 블라인드-지역
# python train_main.py \
# --data_path "./train_data/blind_local.xlsx" \
# --text_col "text" \
# --label_col "label" \
# --config_file_path './config.yml' \
# --config "blind_local"

# # 블라인드-학력
python3 train_main.py \
--data_path "./train_data/blind_schl.xlsx" \
--text_col "text" \
--label_col "label" \
--config_file_path './config.yml' \
--config "blind_schl"
