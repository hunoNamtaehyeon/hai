# Usage
# 1. train

## 1.1. 학습 시작

- 학습 시작 : `sh run_train.sh`
    - 모델 학습은 다음 목록에 대해서만 진행
      - 비속어 검출
      - 타기업 입사지원 검출
      - 블라인드-가족직업 검출
      - 블라인드-성별 검출
      - 블라인드-지역명 검출
      - 블라인드-학력 검출
    - 상관없는 글 검출의 경우에는 AI모델을 사용하지만 학습이 더 필요하지 않은 task임. 
      - 기존에 개인적으로 학습한 오픈소스([https://github.com/dltmddbs100/SimCSE](https://github.com/dltmddbs100/SimCSE))로 학습한 모델 사용
      - 해당 과업에서 사용되는 인공지능 모델은 문장과 문장이 얼마나 유사성을 띄는지를 학습한 모델로써 라벨링 데이터를 추가하여 학습 한다거나 재학습이 더 이상 필요 없음.
      - 해당 모델 및 config 파일은 `./models/nomatter`에 넣어둠
    
- 모델 사용 : 학습 완료한 모델은 `./models`에 생성된다.

    - 초기 다운로드(구글 드라이브)한 학습모델은 `./models/`에 위치시킨다. 

    - AI모델을 사용하는 7개 과업에서는 각 과업별 모델 폴더 속 `./models/과업/best.pt`의 경로를 argument로 받아 사용한다. 

        - `best.pt`는 학습 과정 중 가장 높은 스코어를 달성한 모델
        - `last.pt`는 미리 지정한 학습 종료 시점에 저장된 모델 : 학습을 오래할 수록 좋은 모델은 아니기 때문에 best.pt와의 비교가 필요해보인다. 

    - 단, `./models/nomatter`의 경우에는 학습에 사용한 AI 모델의 아키텍쳐가 달라 `.pt` 형식이 아닌 `.bin` 형식의 모델로 저장되어있으며 아래와 같은 형식으로만 사용하는 것에 주의한다. 

        ```shell
        --weight_path "./models" \
        --test_model_name "nomatter"
        ```

        

        

        


## 2. inference
- 세부 과업 14개에 대한 검출 진행 시작 : `sh run_infer.sh`
- 세부 과업과 그 파일명
  - 타기업 지원자 검출: `apply_other_com.py`
  - 비속어 사용 검출: `bisok.py`
  - 블라인드-가족직업: `blind_fam_job.py`
  - 블라인드-성별: `blind_gender.py`
  - 블라인드-지역: `blind_local.py`
  - 블라인드-학벌: `blind_schl.py`
  - 글자수 & 문장수:	`count_letter_sent.py`
  - 자사명 오기재: `jasamyeong.py`
  - 자기소개서와 상관없는 글: `no_matter.py`
  - 어절표절-full: `pyojeol_full.py`
  - 어절표절-self: `pyojeol_self.py`
  - 의미없는 내용반복 선별: `repeat_content.py`
  - 반복글자: `repeat_letter.py`
  - 자기소개서 문항 반복 선별: `moonhang.py`
  - 문장 반복 위반: `repeat_sent.py`