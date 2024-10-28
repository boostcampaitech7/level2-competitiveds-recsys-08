<p align="center">

  <h1> 💡 수도권 아파트 전세가 예측 프로젝트 </h1>

  > 이 프로젝트는 주어진 데이터를 바탕으로 전세가를 예측하는 AI 알고리즘을 개발하기 위한 프로젝트입니다.
  이를 통해 한국의 전세 시장의 구조와 동향을 이해하고, 인프라와 경제적 요인에 대한 통찰을 기대합니다.

</p>

<br>

## 🏢 Team

### 🗣️ 팀 소개
>**저희 팀은 개발자처럼 협업하고, 체계적으로 가설을 세워 실험하고 기록하며 여러 모델을 경험해보는 것을 목표로 합니다.**


### 👨🏼‍💻 Members
김영찬|박광진|박세연|박재현|배현우|조유솔|
:-:|:-:|:-:|:-:|:-:|:-:
<img src='https://avatars.githubusercontent.com/Sudkorea' height=60 width=60></img>|<img src='https://avatars.githubusercontent.com/CroinDA' height=60 width=60></img>|<img src='https://avatars.githubusercontent.com/SayOny' height=60 width=60></img>|<img src='https://avatars.githubusercontent.com/JaeHyun11' height=60 width=60></img>|<img src='https://avatars.githubusercontent.com/hwbae42' height=60 width=60></img>|<img src='https://avatars.githubusercontent.com/YusolCho' height=60 width=60></img>|
<a href="https://github.com/Sudkorea" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/></a>|<a href="https://github.com/CroinDA" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/></a>|<a href="https://github.com/SayOny" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/></a>|<a href="https://github.com/JaeHyun11" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/></a>|<a href="https://github.com/hwbae42" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/>|<a href="https://github.com/YusolCho" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/></a>

<br>

## 🛠️ 기술 스택 및 협업
  <img src="https://img.shields.io/badge/Python-3776AB?style=square&logo=Python&logoColor=white"/>&nbsp;

  <img src="https://img.shields.io/badge/scikitlearn-F7931E?style=square&logo=scikitlearn&logoColor=white"/>&nbsp;
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white"/>&nbsp;

  <img src="https://img.shields.io/badge/Optuna-3863A0?style=square&logo=Optuna&logoColor=white"/>&nbsp;
  <img src="https://img.shields.io/badge/Pandas-150458?style=square&logo=Pandas&logoColor=white"/>&nbsp;

  <img src="https://img.shields.io/badge/Notion-000000?style=square&logo=Notion&logoColor=white"/>&nbsp;
  <img src="https://img.shields.io/badge/weightsandbiases-FFBE00?style=square&logo=weightsandbiases&logoColor=white"/>&nbsp;

### 🤝 협업 방식
<img width="990" alt="스크린샷 2024-10-26 20 14 43" src="https://github.com/user-attachments/assets/6d892bce-6e4a-4705-8e70-b9b431ce5164">
<img width="982" alt="스크린샷 2024-10-26 20 15 09" src="https://github.com/user-attachments/assets/604ae936-a19a-4583-95d7-7f3f21165d53">
<img width="955" alt="스크린샷 2024-10-26 20 16 02" src="https://github.com/user-attachments/assets/98f392ee-7aa0-4669-aa6a-eda0055820b5">
<img width="961" alt="스크린샷 2024-10-26 20 20 02" src="https://github.com/user-attachments/assets/83fe5143-1ac7-43ea-86f4-7b8859361104">
<img width="944" alt="스크린샷 2024-10-26 20 20 44" src="https://github.com/user-attachments/assets/278052a9-6314-4995-b0c6-0ab35a55ef93">

<br>

## Pipeline
<img width="914" alt="스크린샷 2024-10-26 20 36 00" src="https://github.com/user-attachments/assets/7413a04e-f8f1-4b3d-a54e-48c9650e4d86">

<br>

## 📁 Directory
```bash
project/
│
├── notebooks/
│    ├── EDA.ipynb
│    └── baseline.ipynb
├── docs/
│    └── 랩업리포트, 발표자료 등
├── src/
│    ├── data/
│       ├── __init__.py
│       ├── preprocessor.py
│       └── features.py
│    ├── models/
│       ├── __init__.py
│       ├── ensemble.py
│       ├── lgbm.py     # LGBM 모델
│       ├── xgb.py      # XGBoost 모델
│       └── rf.py # Random Forest 모델
│    ├── arg_paser.py   # 커맨드 라인 옵션 인풋 설정
│    └── utils.py   # 완디비 콜백 커스텀, 모델 저장 및 불러오기
├── configs/
│    └── train_config.yaml
├── saved/
│    └── models/   # 모델 별 pkl 파일 저장
├── train.py
└── test.py


```
<br>

# 🏃 How to run
## 요구사항 Requirements
이 프로젝트를 실행하기 위해 필요한 모든 라이브러리를 설치하려면 다음 명령어를 실행하세요:
```bash
pip install -r requirements.txt
```
## 설정 Configuration
이 프로젝트는 `configs/` 디렉토리 내의 YAML 파일로 모델 하이퍼 파라미터와 기타 설정을 관리합니다.
### 파일 구조

설정 파일(`config.yaml`)은 다음과 같은 주요 섹션으로 구성되어 있습니다:

1. 공통 설정 (common)
2. LightGBM 설정 (lightgbm)
3. CatBoost 설정 (catboost)
4. Random Forest 설정 (rf)

각 섹션에는 해당 모델 또는 전체 프로젝트에 필요한 다양한 매개변수가 포함되어 있습니다.

### 예시

설정 파일의 일부 예시:

```yaml
common:
  data_path: "../../data/"
  random_seed: 42
  n_splits: 5  # KFold split

lightgbm:
  objective: "regression"
  metric: ["mae", "rmse"]
  num_leaves: 1200
  learning_rate: 0.035
  # ...
```
## 학습 및 예측
### Training

개별 모델을 훈련하려면 다음 명령어를 사용하세요:

```bash
python train.py -lgb
python train.py -cat
python train.py -rf
# 또는 여러 모델을 한번에 학습시킬 수도 있습니다
python tarin.py -lgb -cat -rf
```

### Testing

앙상블 모델을 테스트 하려면 다음 명령어를 사용하세요:
```bash
python test.py
```

<br>
