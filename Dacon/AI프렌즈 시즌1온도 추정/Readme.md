## AI프렌즈 시즌1 온도 추정 경진대회

**배경**

우리나라에는 전국에 걸쳐 시도별 기상관측소가 있어 지역별 기온을 알 수 있습니다. 하지만, 각 지역 내에서도 대상과 위치에 따라 온도차이가 매우 많이 납니다. 더운 여름날 뜨거운 아스팔트 위를 걸어보셨거나, 겨울철 칼바람 부는 교량위를 걸어보신 분들은 체감하셨을 겁니다. 그렇다고 '모든 곳'에 관측소를 만들어 '지속적'으로 측정하기란 현실적으로 어렵습니다.

그래서 생각해 낸 방법이 ‘기상청 공공데이터를 활용한 온도추정’ 입니다. 저가의 센서로 관심대상의 온도를 단기간 측정하여 기상청의 관측 데이터와의 상관관계 모델을 만들고, 이후엔 생성된 모델을 통해 온도를 추정하여 서비스하는 것입니다. 2013년 10월부터 시행된 ‘공공데이터의 제공 및 이용에 관한 법률 제 21조’ 에 의해 기상청에서 데이터를 무료로 제공하고 있습니다. 멋지지 않나요? 그럼, 새로운 알고리즘을 통해 '나만의 기상청'을 만들어주세요.



**2. 주최/주관/후원**

\- 주최 : AI프렌즈, 한국원자력연구원, 한국기계연구원, DACON

\- 주관 : DACON

\- 후원 : 연구개발특구진흥재단



# 진행방향

1. EDA를 통해서 데이터 탐색을 진행했었다. 

2. 먼저 33일의 train 데이터셋에서 구하고자 하는 Y18 종속변수는 30일치 동안 NULL 처리 되어있어서 이부분을 중점적으로 Y18센서와 다른 센서간의 상관관계들을 확인하였고 그중 상관관계가 제일 높은 Y15,16 의 평균값을 Y18 대체하였다

3. Train 데이터의 1일치 행의 개수를 통해서 시간과 분 단위로 쪼개서 feature 를 추가하였다.

   ```python
   # ID로 시간 변수 생성
   minute = pd.Series((X_train_norm.index%144).astype(int))
   hour= pd.Series((X_train_norm.index%144/6).astype(int))
   
   # 삼각함수를 이용한 시간변수 생성
   min_in_day = 24*6
   hour_in_day = 24
   
   minute_sin = np.sin(np.pi*minute/min_in_day) 
   minute_cos = np.cos(np.pi*minute/min_in_day)
   
   hour_sin  = np.sin(np.pi*hour/hour_in_day)
   hour_cos  = np.cos(np.pi*hour/hour_in_day)
   
   X_train_norm['minute_sin'] = minute_sin
   X_train_norm['minute_cos'] = minute_cos
   
   X_train_norm['hour_sin'] = hour_sin
   X_train_norm['hour_cos'] = hour_cos
   X_train_norm
   ```

4. 일일강수량 같은 feature 들은 모두 순간강수량으로 바꿔주었다.

```python
def momentValue(df, columns):
    for column in columns:
        temp_df = pd.DataFrame(df[column])
        temp_df['dummy'] = 0
        
        for i in range(1, temp_df.shape[0]):
            a = temp_df[column].iloc[i] - temp_df[column].iloc[i-1]
            if a > 0:
                temp_df['dummy'].iloc[i] = a
        
        df[column] = temp_df['dummy']
    
    return df
```

5. 이후 제출했을때 xgb, lgb 모델을 각각 시도해보았고 같은 하이퍼파라미터를 적용했을때 xgb가 조금더 점수가 좋았다.
6. train 데이터가 test 데이터셋보다 개수가 작고, 주기성을 띄우고 있다고 생각하여 train 데이터 전체를 2배로 늘려줘봤고, 점수가 향상되는것을 확인하였다.
7. 이후 Y18을 예측하는 방법을 다시 생각해서 semi-supervised 를 사용해보고자 해봤다.
8. train 먼저 Y01~Y17을  예측을 해서 3일치 Null 값을 채운후 다시 Y18을 예측 하는 방법이었다.
9. 시도해보았고 결과가 좋지는 않았다.

> 1등 코드를 보면 내가 생각했던 semi-supervised를 사용했는데 그분은 여기에 그림자 까지 계산을 했었다.

10. 그후 모델링에 중점을 두었고 하이퍼 파라미터 튜닝을 하면서 조금씩 점수를 향상시켰다.
11. 이때 KFOLD, baysian, shap 들을 사용했었고 baysian,shap을 통해 importance features 들을 뽑아내서 적용해 점수를 향상시켰었다.
12. 마지막으로 XGB,LGB 두개를 앙상블하면서 제출하고 마무리 하였다.



## 후기

처음 제대로 참가해본 데이터 분석대회다.

마지막 4일전에 20등을 찍어놓고 일이 있어서 2일정도 못했더니 48등까지 내려가 있는 모습을 보고 경악을 받았던게 인상 깊은 대회다 끝내 20등을 복구하지 못해서 안타깝다. 

이번 대회를 하면서 XGB, LGB, shap, baysian 과 같은 모델과 파라미터 튜닝을 알아가게 되었다 그리고 각 모델마다 데이터에 따라서 성능이 다르다는것과 모델마다 성능이 잘 나오는 옵티마이저가 있다는것을 다시 느끼게 되었다.

예로 ) LGB는 KFOLD, SHAP 과 더 잘맞았고 train_test_split, baysian은 별로였다.

그와 반대로 XGB는 train_test_split,baysian과 잘맞았고 , KFOLD,shap과 는 별로였다.


