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
│          ├── __init__.py       
│          ├── data_loader.py       
│          ├── preprocessor.py      
│          ├── features.py       
│    ├── models/          
│       ├── __init__.py       
│       ├── ensemble.py     
│       ├── lgbm.py     # LGBM 모델
│       ├── xgb.py      # XGBoost 모델
│       └── catboost.py # CatBoost 모델
│       ├── arg_paser.py   # 커맨드 라인 옵션 인풋 설정
│       └── utils.py   # 완디비 콜백 커스텀, 모델 저장 및 불러오기
├── configs/ 
│    ├── train_config.yaml      
│    └── test_config.yaml      
├── saved/  
│    ├── models/   # 모델 별 pkl 파일 저장         
│    └── logs/                    
├── train.py      
└── test.py


```

## 🏃 How to run

### Training

모델을 훈련하려면 다음 명령어를 사용하세요:

```bash
python train.py -lgb
python train.py -cat
python train.py -rf
```

### Testing

모델을 테스트 하려면 다음 명령어를 사용하세요:
```bash
python test.py
```

<br>
