# Task153  미세 Crack 검출 모델 [태영세라믹] (Team. 놀면 뭐하늬)



사용모델 : efficient 6,7

제출 파일 : (EFF6, EFF7,predict).ipynb , (Best_EFN_B6_ensemble, Best_EFN_B7_V5).h5, crack_final_95.csv

 

## 코드 설명

1. f1-score Matrix 구하는 함수입니다

```python
def recall(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
    count_true_positive_false_negative = K.sum(y_target_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    # return a single tensor value
    return recall


def precision(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    # (True Positive + False Positive) = 예측 값이 1(Positive) 전체
    count_true_positive_false_positive = K.sum(y_pred_yn)

    # Precision = (True Positive) / (True Positive + False Positive)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())

    # return a single tensor value
    return precision


def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    _f1score = ( 2 * _recall * _precision) / (_recall + _precision+ K.epsilon())
    
    # return a single tensor value
    return _f1score
```

2. image를 불러오고 전처리하는 과정을 포함한 함수 입니다

```python
def read_image(image_path, resize_ratio=2):
    if not(isinstance(image_path, str)):
        # if tensor with byte string
        image_path = image_path.numpy().decode('utf-8')

    image_level_1 = skimage.io.MultiImage(image_path)[0]
    if resize_ratio != 1:
        new_w = int(image_level_1.shape[1]*resize_ratio)
        new_h = int(image_level_1.shape[0]*resize_ratio)
        image_level_1 = cv2.resize(
            image_level_1, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image_level_1
```

3. modeling(이미지 사이즈를 2배로 늘려 136,136으로 input size 설정 합니다.) 

```python
def build_lrfn(lr_start=0.00001, lr_max=0.00005,lr_min=0.00001, lr_rampup_epochs=5,lr_sustain_epochs=0, lr_exp_decay=.8):
    
#     lr_max = lr_max * strategy.num_replicas_in_sync
    lr_max = lr_max * 8
    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) *\
                 lr_exp_decay**(epoch - lr_rampup_epochs- lr_sustain_epochs) + lr_min
        return lr
    return lrfn

def Eff_B7():
    model = Sequential([efn.EfficientNetB7(input_shape=(136,136,3),weights='noisy-student',include_top=False),
                                 tf.keras.layers.GlobalAveragePooling2D(),
                                 tf.keras.layers.BatchNormalization(),
                                 tf.keras.layers.Dense(128,activation='relu',kernel_initializer='he_normal'),
                                 tf.keras.layers.Dense(64,activation='relu',kernel_initializer='he_normal'),
                                 tf.keras.layers.Dropout(0.5),
                                 tf.keras.layers.Dense(2,activation='softmax',kernel_initializer='he_normal')])               
    model.compile(optimizer=Adam(lr=0.001, epsilon=0.001,decay=1e-5, clipnorm=1.),loss = 'categorical_crossentropy',metrics=['categorical_accuracy',precision, recall, f1score])
    
    
    return model
def Eff_B6():
    model = Sequential([efn.EfficientNetB6(input_shape=(136,136,3),weights='noisy-student',include_top=False),
                                 tf.keras.layers.GlobalAveragePooling2D(),
                                 tf.keras.layers.BatchNormalization(),
                                 tf.keras.layers.Dense(128,activation='relu',kernel_initializer='he_normal'),
                                 tf.keras.layers.Dense(64,activation='relu',kernel_initializer='he_normal'),
                                 tf.keras.layers.Dropout(0.5),
                                 tf.keras.layers.Dense(2,activation='softmax',kernel_initializer='he_normal')])               
    model.compile(optimizer=Adam(lr=0.001, epsilon=0.001,decay=1e-5, clipnorm=1.),loss = 'categorical_crossentropy',metrics=['categorical_accuracy',precision, recall, f1score])
    
    
    return model
model_95=Eff_B7()
model_93=Eff_B6()
```

4. model save 및 callyback, schedule 등 설정하는 부분입니다.

```text
date = '20201114'
model = 'EFN_B7'
version = 'V5_resize'
batch_size = 30
epochs = 50
model_file ='Best_' + model +'_'+ version+'.h5'
lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
EarlyStopping=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=5,verbose=True, mode="min")
modelsaver = tf.keras.callbacks.ModelCheckpoint(
    model_file, 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True,
    mode='min'
)
```

5. KFOLD 를 사용하여 모델 train (모델 실행시 많은 다수의 h5가 생성되는데 그중 model_file ='Best_' + model +'_'+ version+'.h5' 이 h5 활용했습니다.)

```python
kf = KFold(n_splits=4)
index=1
for fold,(train,val) in enumerate(kf.split(df)):
    print('Fold:', index)
    train_df = df.iloc[train]
    val_df = df.iloc[val]
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    train_N = train_df.shape[0]
    train_x = np.empty((train_N, 136,136,3), dtype=np.uint8)
    for i, path in enumerate(tqdm(train_df.file)):
        image = (read_image(task_dir+path))
        train_x[i,:,:,:] =  image
    train_y = to_categorical(train_df['label'])
    
    val_N = val_df.shape[0]
    val_x = np.empty((val_N, 136,136,3), dtype=np.uint8)
    for i, path in enumerate(tqdm(val_df.file)):
        image = (read_image(task_dir+path))
        val_x[i,:,:,:] = image
    val_y = to_categorical(val_df['label'])
    
    model = model_Eff.fit(train_x,
                            train_y,
                            epochs=epochs,
                            validation_data = (val_x, val_y),
                            verbose=2,
                            steps_per_epoch=train.shape[0]//batch_size,
                        callbacks=[lr_schedule,EarlyStopping,modelsaver])
    
    model_Eff.save('./'+date+'/'+model_file + '_'+str(index))
    index+=1

model_Eff.save('./'+'EFF'+version+ '.h5')
```

6. test file load (train 과 동일하게 load 합니다.)

```python
test_PATH = '../../data/.train/.task153/data/test/'
df_T = pd.read_csv(test_PATH+'test.csv')
test_N = df_T.shape[0]
test_x = np.empty((test_N,136,136,3), dtype=np.uint8)
for i,path in enumerate(tqdm(df_T.file_name)): 
    image = read_image(test_PATH + path)
    test_x[i,:,:,:] =  image
```

7. best model 불러오기 (metrix를 지정해줘서 load 할때 함수를 지정 합니다.)

```python
save_model = keras.models.load_model("./Best_EFN_B6_ensemble.h5",
                                              custom_objects={
                                                  'recall': recall,
                                                  'precision': precision,
                                                  'f1score' : f1score}
                                             )
save_model2 = keras.models.load_model("./Best_EFN_B7_V5_resize.h5",
                                              custom_objects={
                                                  'recall': recall,
                                                  'precision': precision,
                                                  'f1score' : f1score}
                                             )
```

8. load model predict(모델 예측 및 csv 저장 합니다.)

```python
preds_a = save_model.predict(test_x)
preds_b = save_model2.predict(test_x)
test_df['label'] = np.argmax(preds_a*0.8+preds_b * 0.2,axis=1)
test_df.label.to_csv('/home/workspace/user-workspace/prediction/ensemble.csv',header=None,index=None)
```

