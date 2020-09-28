# LG 블럭공정



# 점수 향상 방법

1. 수요 초과분 외 다양한 양상을 반영하는 목적함수.
2. 활성화 함수 변경
3. 23일 이전 MOL 예측 값 포함.
4. 월별 최대 MOL 투입량 반영.
5. 성형 공정 2개 라인을 따로 모델링.

# 베이스라인

* simulator.py
* genome.py
* main.py

기본적으로 EVENT 와 MOL은 genome.py를 통해 예측하고, PRT는 simulator.py를 통해 자동적으로 계산됩니다.

[이 글](https://dacon.io/competitions/official/235612/talkboard/400906?page=2&dtype=recent&ptype=pub)에서 확인 할 수 있듯이, PRT는 제약이 없습니다. 또한 simulator.py를 통해 MOL에서 역산으로 구해지기 때문에 신경 쓸 필요가 없습니다.



## 1.1 simulator.py

simulator.py 파일의 수정 방향은 점수 향상 방법의 1번 항목입니다.

### 1.1.1) 수요 초과분 외 다양한 양상을 반영하는 목적함수
> [[80점 넘기는 팁, 리워드 수정 by 우리집전자렌지골드스타](https://dacon.io/competitions/official/235612/codeshare/1142?page=1&dtype=recent&ptype=pub)]  
> [[시뮬레이터 개선 by 행복한금요일](https://dacon.io/competitions/official/235612/codeshare/1221?page=1&dtype=recent&ptype=pub)]

위 목적을 달성하기 위해서 `cal_score()`함수만 수정하면 됩니다. 수정에는 [첫번째 글](https://dacon.io/competitions/official/235612/codeshare/1142?page=1&dtype=recent&ptype=pub) 글을 거의 그대로 사용했습니다.

```python
def cal_score(self, blk_diffs):
        # Block Order Difference
        blk_diff_m = 0
        blk_diff_p = 0
        for item in blk_diffs:
            if item < 0:
                blk_diff_m = blk_diff_m + abs(item)
            if item > 0:
                blk_diff_p = blk_diff_p + abs(item)
        score = 5*blk_diff_m + 2*blk_diff_p
        return score
```

다만 [평가식](https://dacon.io/competitions/official/235612/overview/)에서 확인할 수 있듯, 초과분과 수요분에 대한 비중이 다릅니다. 따라서 비중은 학습을 위해 반영해 주는 것이 좋다고 생각합니다.



## 1.2 genome.py

점수 향상 방법의 나머지 항목은 실질적으로 genome.py에 반영됩니다.

### 1.2.1) 활성화 함수 변경:

> [MOL 수량에 관한 코멘트](https://dacon.io/competitions/official/235612/talkboard/400835?page=1&dtype=recent&ptype=pub)

베이스라인 genome.py의 `foward()` 함수 내 MOL 수량 신경망을 살펴보면 마지막에 activation fuction으로 softmax를 사용하고 있으며, 그 후 `argmax()` 함수를 사용하고 있습니다. 그리고 이는 main.py의 4. 변수 선택 및 모델 구축에 `output_length_2 = 12 # MOL(0~5.5, step:0.5)`와 연관이 있습니다.

결론만 이야기하면, MOL 수량 예측시 0, 0.5 1, ... , 5, 5.5 총 12개 중에서 골라 하나의 값으로 예측한다는 의미입니다. 베이스라인을 실행시켜보시면 아시겠지만, 생성된 파일에 MOL_A, B 컬럼에는 12개의 값으로만 채워져있을 것입니다.

따라서 이를 개선하기 위해 활성화 함수를 sigmoid를 사용하고 `argmax()` 라인을 삭제해서 아래와 같이 MOL 수량 예측값이 실수형이 될 수 있도록 수정하면 됩니다.(인용글의 6월 4일자 Dacon의 답변 참고)

```python
# net = self.softmax(net)
# out2 = np.argmax(net)
# out2 /= 2
# 위 주석 처리한 코드를 아래와 같이 변경 #

net = self.sigmoid(net)
out2 = net
```

활성화 함수로 sigmoid를 사용한 이유는 예측값의 범위를 지정하기 위해서입니다. MOL 수량은 음수여서는 안되며, 월별로 최대 투입량 한계가 존재합니다. 따라서 sigmoid 함수 통과한 예측값은 (0, 1) 사이의 값만 가지기 때문에 후에 서술할 월별 최대 MOL 투입량을 반영하여 개선할 수 있습니다. __(다만 이 글을 작성하면서 살펴보니 softmax의 범위도 (0, 1)이었습니다. 둘 중 어떤 함수가 더 효과적인지에 대한 지식이나 테스트는 없습니다.)__



### 1.2.2) 월별 최대 MOL 투입량 반영

위에서 말씀드린 베이스라인의 main.py 0~5.5까지의 값에서 5.5라는 최대값은 는 월별 최대 MOL 투입량을 보수적으로 접근한 것이라고 생각합니다. 즉 max_count.csv의 최소값인 약 5.8 (=140.5907407/24)을 기준으로 설정된 값입니다.

따라서 sigmoid 함수와 더불어 max_count.csv의 월별 최대 MOL 투입량을 반영하면 더욱 정확한 예측이 가능합니다.

```python
self.submission.loc[s, 'Event_A'] = out1
if self.submission.loc[s, 'Event_A'] == 'PROCESS':
    # self.submission.loc[s, 'MOL_A'] = out2
    if s < 24*14:
        self.submission.loc[s, 'MOL_A'] = 0
    elif 24*14 <= s <24*23:
        if self.process_mode_A == 1:
            self.submission.loc[s, 'MOL_A'] = out2 * (140.5907407/24)
        else:
            self.submission.loc[s, 'MOL_A'] = 0
    elif 24*23 <= s < 24*30:
        self.submission.loc[s, 'MOL_A'] = out2 * (140.5907407/24)
    elif 24*30 <= s < 24*61:
        self.submission.loc[s, 'MOL_A'] = out2 * (140.8055556/24)
    else:
        self.submission.loc[s, 'MOL_A'] = out2 * (141.0185185/24)
else:
    self.submission.loc[s, 'MOL_A'] = 0
```

14일부터 23일까지의 코드에 대해서는 아래에서 마저 말씀드리겠습니다.



### 1.2.3) 23일 이전 MOL 예측 값 포함

Baseline 152, 1533번째 줄을 보면 아래와 같은 코드가 있습니다.

```python
# 23일간 MOL = 0
self.submission.loc[:24*23, 'MOL_A'] = 0
```

이 코드는 PRT 생산에 23일이 소요되기 때문에 4월 1일 0시 부터 PRT를 생산해도 4월 24일 0시 부터 MOL을 생산할 수 있기 때문에 적용된 코드라고 생각합니다. simulator.py에서 MOL 수량을 기준으로 PRT를 역산하기 때문에 23일 이전에 투입되는 MOL이 있다면 역산으로 PRT를 구하는 것이 논리적으로 말이 안되기 떄문입니다.

그러나 stock.csv 파일에서 확인할 수 있듯이 PRT, MOL 일부 재고가 존재하기 때문에, 이를 소모할 수 있다면 점수가 더욱 향상될 것이라고 판단하였습니다.

다만 주의해야 하는 점이 MOL 생산은 14일부터만 가능하다는 것입니다.

이런 상황을 반영한 것이 월별 최대 MOL 투입량을 반영한 위 코드에서 23일까지의 코드입니다. 위와 같이 수정했다면 `self.submission.loc[:24*23, 'MOL_A'] = 0`는 주석처리 하시면 됩니다.

`if self.process_mode_A == 1`는 MOL_2가 생산될 수 있는 PROCESS 일때만 값을 넣는 것인데 stock.csv에서 확인할 수 있듯 재고에는 PRT2 밖에 없으므로 process_mode_A가 1일 때만 예측값을 사용했습니다.



### 1.2.4) 성형 공정 2개 라인을 따로 모델링

따로 모델링한다고 해서 특별한 것은 없습니다. 그냥 복붙으로 out1, 2에서 out 1, 2, 3, 4로 늘려주시면 됩니다. 그에 맞춰 `update_mask()`, `forward()`, `predict()` 함수 및 다른 함수들도 수정해 주시면 됩니다. 전체 코드에서 확인하시면 됩니다.
