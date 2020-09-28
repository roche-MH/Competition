import os
import pandas as pd
import numpy as np
from pathlib import Path
from module.simulator import Simulator
simulator = Simulator()
submission_ini = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'sample_submission.csv'))
order_ini = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'order.csv'))

class Genome():
    def __init__(self, score_ini, input_len, output_len_1, output_len_2, h1=50, h2=50, h3=50):
        # 평가 점수 초기화
        self.score = score_ini
        
        # 히든레이어 노드 개수
        self.hidden_layer1 = h1
        self.hidden_layer2 = h2
        self.hidden_layer3 = h3
        
        # Event 신경망 가중치 생성
        self.w1 = np.random.randn(input_len, self.hidden_layer1)
        self.w2 = np.random.randn(self.hidden_layer1, self.hidden_layer2)
        self.w3 = np.random.randn(self.hidden_layer2, self.hidden_layer3)
        self.w4 = np.random.randn(self.hidden_layer3, output_len_1)
        
        # MOL 수량 신경망 가중치 생성
        self.w5 = np.random.randn(input_len, self.hidden_layer1)
        self.w6 = np.random.randn(self.hidden_layer1, self.hidden_layer2)
        self.w7 = np.random.randn(self.hidden_layer2, self.hidden_layer3)
        self.w8 = np.random.randn(self.hidden_layer3, output_len_2)

        # Event_B 신경망 가중치 생성
        self.w9 = np.random.randn(input_len, self.hidden_layer1)
        self.w10 = np.random.randn(self.hidden_layer1, self.hidden_layer2)
        self.w11 = np.random.randn(self.hidden_layer2, self.hidden_layer3)
        self.w12 = np.random.randn(self.hidden_layer3, output_len_1)
        
        # MOL_B 수량 신경망 가중치 생성
        self.w13 = np.random.randn(input_len, self.hidden_layer1)
        self.w14 = np.random.randn(self.hidden_layer1, self.hidden_layer2)
        self.w15 = np.random.randn(self.hidden_layer2, self.hidden_layer3)
        self.w16 = np.random.randn(self.hidden_layer3, output_len_2)       
        
        # Event 종류
        self.mask_A = np.zeros([5], np.bool) # 가능한 이벤트 검사용 마스크
        self.event_map = {0:'CHECK_1', 1:'CHECK_2', 2:'CHECK_3', 3:'CHECK_4', 4:'PROCESS'}
        
        self.check_time_A = 28    # 28시간 검사를 완료했는지 검사, CHECK Event시 -1, processtime_time >=98 이면 28
        self.process_A = 0        # 생산 가능 여부, 0 이면 28 시간 검사 필요
        self.process_mode_A = 0   # 생산 물품 번호 1~4, stop시 0
        self.process_time_A = 0   # 생산시간이 얼마나 지속되었는지 검사, PROCESS +1, CHANGE +1, 최대 140

        # Event_B 종류
        self.mask_B = np.zeros([5], np.bool) # 가능한 이벤트 검사용 마스크
        # self.event_map = {0:'CHECK_1', 1:'CHECK_2', 2:'CHECK_3', 3:'CHECK_4', 4:'PROCESS'}
        
        self.check_time_B = 28    # 28시간 검사를 완료했는지 검사, CHECK Event시 -1, processtime_time >=98 이면 28
        self.process_B = 0        # 생산 가능 여부, 0 이면 28 시간 검사 필요
        self.process_mode_B = 0   # 생산 물품 번호 1~4, stop시 0
        self.process_time_B = 0   # 생산시간이 얼마나 지속되었는지 검사, PROCESS +1, CHANGE +1, 최대 140
    
    def update_mask_A(self):
        self.mask_A[:] = False
        if self.process_A == 0: # 생산이 불가능하면, 28시간 검사 필요.
            if self.check_time_A == 28: # 28시간 검사 통과했으면
                self.mask_A[:4] = True # CHECK_1~4까지 True
            if self.check_time_A < 28: # 28시간 검사 통과 못했으면
                self.mask_A[self.process_mode_A] = True # 
        if self.process_A == 1: # 생산 가능
            self.mask_A[4] = True
            if self.process_time_A > 98:
                self.mask_A[:4] = True

    def update_mask_B(self):
        self.mask_B[:] = False
        if self.process_B == 0: # 생산이 불가능하면, 28시간 검사 필요.
            if self.check_time_B == 28: # 28시간 검사 통과했으면
                self.mask_B[:4] = True # CHECK_1~4까지 True
            if self.check_time_B < 28: # 28시간 검사 통과 못했으면
                self.mask_B[self.process_mode_B] = True # 
        if self.process_B == 1: # 생산 가능
            self.mask_B[4] = True
            if self.process_time_B > 98:
                self.mask_B[:4] = True
    
    def forward(self, inputs):
        # Event 신경망
        net = np.matmul(inputs, self.w1)
        net = self.linear(net)
        net = np.matmul(net, self.w2)
        net = self.linear(net)
        net = np.matmul(net, self.w3)
        net = self.sigmoid(net)
        net = np.matmul(net, self.w4)
        net = self.softmax(net)
        net += 1
        net = net * self.mask_A
        out1 = self.event_map[np.argmax(net)]
        
        # MOL 수량 신경망
        net = np.matmul(inputs, self.w5)
        net = self.linear(net)
        net = np.matmul(net, self.w6)
        net = self.linear(net)
        net = np.matmul(net, self.w7)
        net = self.sigmoid(net)
        net = np.matmul(net, self.w8)
        # net = self.softmax(net)
        # out2 = np.argmax(net)
        # out2 /= 2
        net = self.sigmoid(net)
        out2 = net

        # Event_B (Line_B) 신경망
        net = np.matmul(inputs, self.w9)
        net = self.linear(net)
        net = np.matmul(net, self.w10)
        net = self.linear(net)
        net = np.matmul(net, self.w11)
        net = self.sigmoid(net)
        net = np.matmul(net, self.w12)
        net = self.softmax(net)
        net += 1
        net = net * self.mask_B
        out3 = self.event_map[np.argmax(net)] # out3은 Event_B를 출력
        
        # MOL_B (Line_B) 수량 신경망
        net = np.matmul(inputs, self.w13)
        net = self.linear(net)
        net = np.matmul(net, self.w14)
        net = self.linear(net)
        net = np.matmul(net, self.w15)
        net = self.sigmoid(net)
        net = np.matmul(net, self.w16)
        # net = self.softmax(net)
        # out4 = np.argmax(net) # 출력이 12개이니까 0~11까지 중에 하나 선택
        # out4 /= 2 # 시간당 최대 투입개수가 140/24 개 니까 최대 5.5개 까지라고 생각. u know what im saying
        net = self.sigmoid(net)
        out4 = net

        return out1, out2, out3, out4

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def linear(self, x):
        return x
    
    def create_order(self, order):
        for i in range(30):
            order.loc[91+i,:] = ['0000-00-00', 0, 0, 0, 0]        
        return order
   
    def predict(self, order):
        order = self.create_order(order)
        self.submission = submission_ini
        self.submission.loc[:, 'PRT_1':'PRT_4'] = 0
        for s in range(self.submission.shape[0]):
            self.update_mask_A()
            self.update_mask_B()
            inputs = np.array(order.loc[s//24:(s//24+30), 'BLK_1':'BLK_4']).reshape(-1)
            inputs = np.append(inputs, s%24)
            out1, out2, out3, out4 = self.forward(inputs)
            
            if out1 == 'CHECK_1':
                if self.process_A == 1:
                    self.process_A = 0
                    self.check_time_A = 28
                self.check_time_A -= 1
                self.process_mode_A = 0
                if self.check_time_A == 0:
                    self.process_A = 1
                    self.process_time_A = 0
            elif out1 == 'CHECK_2':
                if self.process_A == 1:
                    self.process_A = 0
                    self.check_time_A = 28
                self.check_time_A -= 1
                self.process_mode_A = 1
                if self.check_time_A == 0:
                    self.process_A =1
                    self.process_time_A = 0
            elif out1 == 'CHECK_3':
                if self.process_A == 1:
                    self.process_A = 0
                    self.check_time_A = 28
                self.check_time_A -= 1
                self.process_mode_A = 2
                if self.check_time_A == 0:
                    self.process_A = 1
                    self.process_time_A = 0
            elif out1 == 'CHECK_4':
                if self.process_A == 1:
                    self.process_A = 0
                    self.check_time_A = 28
                self.check_time_A -= 1
                self.process_mode_A = 3
                if self.check_time_A == 0:
                    self.process_A = 1
                    self.process_time_A = 0
            elif out1 == 'PROCESS':
                self.process_time_A += 1
                if self.process_time_A == 140:
                    self.process_A = 0
                    self.check_time_A = 28
                    
            if out3 == 'CHECK_1':
                if self.process_B == 1:
                    self.process_B = 0
                    self.check_time_B = 28
                self.check_time_B -= 1
                self.process_mode_B = 0
                if self.check_time_B == 0:
                    self.process_B = 1
                    self.process_time_B = 0
            elif out3 == 'CHECK_2':
                if self.process_B == 1:
                    self.process_B = 0
                    self.check_time_B = 28
                self.check_time_B -= 1
                self.process_mode_B = 1
                if self.check_time_B == 0:
                    self.process_B =1
                    self.process_time_B = 0
            elif out3 == 'CHECK_3':
                if self.process_B == 1:
                    self.process_B = 0
                    self.check_time_B = 28
                self.check_time_B -= 1
                self.process_mode_B = 2
                if self.check_time_B == 0:
                    self.process_B = 1
                    self.process_time_B = 0
            elif out3 == 'CHECK_4':
                if self.process_B == 1:
                    self.process_B = 0
                    self.check_time_B = 28
                self.check_time_B -= 1
                self.process_mode_B = 3
                if self.check_time_B == 0:
                    self.process_B = 1
                    self.process_time_B = 0
            elif out3 == 'PROCESS':
                self.process_time_B += 1
                if self.process_time_B == 140:
                    self.process_B = 0
                    self.check_time_B = 28

            self.submission.loc[s, 'Event_A'] = out1
            if self.submission.loc[s, 'Event_A'] == 'PROCESS':
                # 아래 코드 사용 시, 위 foward에 out2 수정 요
                # if s < 24*15: 15일 전 조건 이렇게 쓰던가 아니면 아래 23일간 코드 고쳐 쓰던가
                #     self.submission.loc[s, 'MOL_A'] = 0
                # elif 24*15 <= s < 24*30:
                #     self.submission.loc[s, 'MOL_A'] = out2 * (140.5907407/24)
                # elif 24*30 <= s < 24*61:
                #     self.submission.loc[s, 'MOL_A'] = out2 * (140.8055556/24)
                # else:
                #     self.submission.loc[s, 'MOL_A'] = out2 * (141.0185185/24)
                self.submission.loc[s, 'MOL_A'] = out2
            else:
                self.submission.loc[s, 'MOL_A'] = 0

            self.submission.loc[s, 'Event_B'] = out3
            if self.submission.loc[s, 'Event_B'] == 'PROCESS':
                # 아래 코드 사용 시, 위 foward에 out2 수정 요
                # if s < 24*15:
                #     self.submission.loc[s, 'MOL_B'] = 0
                # elif 24*15 <= s < 24*30:
                #     self.submission.loc[s, 'MOL_B'] = out2 * (140.5907407/24)
                # elif 24*30 <= s < 24*61:
                #     self.submission.loc[s, 'MOL_B'] = out2 * (140.8055556/24)
                # else:
                #     self.submission.loc[s, 'MOL_B'] = out2 * (141.0185185/24)
                self.submission.loc[s, 'MOL_B'] = out4
            else:
                self.submission.loc[s, 'MOL_B'] = 0
                
        # 23일간 MOL = 0
        self.submission.loc[:24*23, 'MOL_A'] = 0
        self.submission.loc[:24*23, 'MOL_B'] = 0
        
        # A 라인 = B 라인
        # self.submission.loc[:, 'Event_B'] = self.submission.loc[:, 'Event_A']
        # self.submission.loc[:, 'MOL_B'] = self.submission.loc[:, 'MOL_A']
        
        # 변수 초기화
        self.check_time_A = 28
        self.process_A = 0
        self.process_mode_A = 0
        self.process_time_A = 0

        self.check_time_B = 28
        self.process_B = 0
        self.process_mode_B = 0
        self.process_time_B = 0
        
        return self.submission    
    
def genome_score(genome):
    submission = genome.predict(order_ini)    
    genome.submission = submission    
    genome.score, _ = simulator.get_score(submission)    
    return genome