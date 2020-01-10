import tensorflow as tf
import numpy as np
import pandas as pd
import os, sys

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]='0'

def score_function(predict_label, actual_label):
    
    predict = pd.read_csv(predict_label, engine='python') # 예측 답안 파일 불러오기
    actual = pd.read_csv(actual_label,engine='python') # 실제 답안 파일 불러오기 

    predict.acc_id = predict.acc_id.astype('int')
    predict = predict.sort_values(by =['acc_id'], axis = 0) # 예측 답안을 acc_id 기준으로 정렬 
    predict = predict.reset_index(drop = True)
    actual.acc_id = actual.acc_id.astype('int')
    actual = actual.sort_values(by =['acc_id'], axis = 0) # 실제 답안을 acc_id 기준으로 정렬
    actual =actual.reset_index(drop=True)
    
    if predict.acc_id.equals(actual.acc_id) == False:
        print('acc_id of predicted and actual label does not match')
        sys.exit() # 예측 답안의 acc_id와 실제 답안의 acc_id가 다른 경우 에러처리 
    else:
            
        S, alpha, L, sigma = 30, 0.01, 0.1, 15  
        cost, gamma, add_rev = 0,0,0 
        profit_result = []
        survival_time_pred = list(predict.survival_time)
        amount_spent_pred = list(predict.amount_spent)
        survival_time_actual = list(actual.survival_time)
        amount_spent_actual = list(actual.amount_spent)    
        for i in range(len(survival_time_pred)):
            if survival_time_pred[i] == 64 :                 
                cost = 0
                optimal_cost = 0
            else:
                cost = alpha * S * amount_spent_pred[i]                    #비용 계산
                optimal_cost = alpha * S * amount_spent_actual[i]          #적정비용 계산 
            
            if optimal_cost == 0:
                gamma = 0
            elif cost / optimal_cost < L:
                gamma = 0
            elif cost / optimal_cost >= 1:
                gamma = 1
            else:
                gamma = (cost)/((1-L)*optimal_cost) - L/(1-L)              #반응률 계산
            
            if survival_time_pred[i] == 64 or survival_time_actual[i] == 64:
                T_k = 0
            else:
                T_k = S * np.exp(-((survival_time_pred[i] - survival_time_actual[i])**2)/(2*(sigma)**2))    #추가 생존기간 계산
                
            add_rev = T_k * amount_spent_actual[i]                         #잔존가치 계산
    
           
            profit = gamma * add_rev - cost                                #유저별 기대이익 계산
            profit_result.append(profit)
            
        score = sum(profit_result)                                         #기대이익 총합 계산
    return score


X = np.loadtxt('../preprocess/train_preprocess_1.csv', delimiter=',')
test1 = np.loadtxt('../preprocess/test1_preprocess_1.csv', delimiter=',')
test2 = np.loadtxt('../preprocess/test2_preprocess_1.csv', delimiter=',')

Y = np.loadtxt('../preprocess/train_label_1.csv', delimiter=',')

x = X[:35000, 1:]
x_origin = X[:35000, 1:]
y_day = Y[:35000, 1].reshape(-1,1)
y_money = Y[:35000, 2].reshape(-1,1)

dev_x = X[35000:, 1:]
dev_y_day = Y[35000:, 1].reshape(-1,1)
dev_y_money = Y[35000:, 2].reshape(-1,1)

result_1 = test1[:,:3]
result_2 = test2[:,:3]
test1 = test1[:,1:]
test2 = test2[:,1:]

np.savetxt('./temp/dev_true.csv', Y[35000:,:], delimiter=',', header='acc_id,survival_time,amount_spent', comments='')
np.savetxt('./temp/train_true.csv', Y[:35000,:], delimiter=',', header='acc_id,survival_time,amount_spent', comments='') 


tf.reset_default_graph()

tf.set_random_seed(123)
np.random.seed(123)

X = tf.placeholder(tf.float32, [None, 36])
Y_day = tf.placeholder(tf.float32, [None, 1])
Y_money = tf.placeholder(tf.float32, [None, 1])
keep_prob = tf.placeholder(tf.float32, [])

layer = tf.layers.dense(X, 512, activation='relu')
layer = tf.layers.dense(layer, 256, activation='relu')
layer = tf.layers.dense(layer, 256, activation='relu')
layer = tf.nn.dropout(layer, keep_prob)


day_layer = tf.layers.dense(layer, 128, activation='relu')
day_layer = tf.layers.dense(day_layer, 64, activation='relu')
day_layer = tf.nn.dropout(day_layer, keep_prob)

money_layer = tf.layers.dense(layer, 128, activation='relu')
money_layer = tf.layers.dense(money_layer, 64, activation='relu')
money_layer = tf.nn.dropout(money_layer, keep_prob)

day = tf.layers.dense(day_layer, 1) # , activation='relu'
day = tf.clip_by_value(day, 0, 64)
money = tf.layers.dense(money_layer, 1) # , activation='relu'
money = tf.clip_by_value(money, 0, 40)



cost = -30.*(tf.exp(-tf.pow(day - Y_day, 2)/450) * (tf.nn.sigmoid(money - Y_money/2))*Y_money - 0.01*money)
cost = tf.reduce_mean(cost)

optimizer =  tf.train.AdamOptimizer(0.0002).minimize(cost) #0.00001

index = np.arange(len(x))

SAVER_DIR = "model"
saver = tf.train.Saver()
checkpoint_path = os.path.join(SAVER_DIR, "model")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

max_val = 0
max_all = 0

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    n_epochs = 500
    batch_size = 500
    
    for epoch_i in range(n_epochs):
        np.random.shuffle(index)
        x, y_day, y_money = x[index], y_day[index], y_money[index]
        step = len(x) // batch_size
        
        for i in range(step):
            step_cost = 0
            xs = x[i*batch_size:(i+1)*batch_size]
            ys_money = y_money[i*batch_size:(i+1)*batch_size]
            ys_day = y_day[i*batch_size:(i+1)*batch_size].reshape(-1,1)
            _, train_cost = sess.run([optimizer, cost], feed_dict={X: xs,  Y_money: ys_money, Y_day:ys_day,  keep_prob:0.5})
            step_cost += train_cost
    

        if epoch_i % 1 == 0:
            

            print('epoch :', epoch_i, 'train_cost:', step_cost)

            day_predict = sess.run(day, feed_dict={X: dev_x, keep_prob:1.0})
            money_predict = sess.run(money, feed_dict={X: dev_x, keep_prob:1.0})
            money_predict[money_predict < 0] = 0
            tmp = Y[35000:,:].copy()
            tmp[:, 1] = np.round(day_predict).reshape(-1)
            tmp[:, 2] = money_predict.reshape(-1)
            tmp[:,1] = np.where(tmp[:, 1] > 64, 64, tmp[:, 1])
            tmp[:,1] = np.where(tmp[:, 1] < 0, 0, tmp[:, 1])
            
            
            path = './temp/dev_predict_' + str(epoch_i) + '.csv'
            np.savetxt(path, tmp, delimiter=',', header='acc_id,survival_time,amount_spent', comments='')
            val_score = score_function(path, './temp/dev_true.csv')
            print('dev score:', val_score)
            max_val = max(max_val, val_score)

            day_predict = sess.run(day, feed_dict={X: x_origin, keep_prob:1.0})
            money_predict = sess.run(money, feed_dict={X: x_origin, keep_prob:1.0})
            money_predict[money_predict < 0] = 0
            a = Y[:35000, :].copy()
            a[:, 1] = np.round(day_predict).reshape(-1)
            a[:, 2] = money_predict.reshape(-1)
            a[:,1] = np.where(a[:, 1] > 64, 64, a[:, 1])
            a[:,1] = np.where(a[:, 1] < 0, 0, a[:, 1])
            
            
            path = './temp/train_predict_' + str(epoch_i) + '.csv'
            np.savetxt(path, a, delimiter=',', header='acc_id,survival_time,amount_spent', comments='')
            all_score = score_function(path, './temp/train_true.csv')
            print('train score:', all_score)
            max_all = max(max_all, all_score)

            if (val_score == max_val) or (max_all == all_score):      
                saver.save(sess, checkpoint_path, global_step=epoch_i)

            #############################################################################################################


            day_predict = sess.run(day, feed_dict={X: test1, keep_prob:1.0})
            money_predict = sess.run(money, feed_dict={X: test1, keep_prob:1.0})
            money_predict[money_predict < 0] = 0

            a = result_1.copy()
            a[:, 1] = np.round(day_predict).reshape(-1)
            a[:, 2] = money_predict.reshape(-1)
            a[:,1] = np.where(a[:, 1] > 64, 64, a[:, 1])
            a[:,1] = np.where(a[:, 1] < 0, 0, a[:, 1])
            
            path = './temp/test1_predict_' + str(epoch_i) + '.csv'
            np.savetxt(path, a, delimiter=',', header='acc_id,survival_time,amount_spent', comments='')

            day_predict = sess.run(day, feed_dict={X: test2, keep_prob:1.0})
            money_predict = sess.run(money, feed_dict={X: test2, keep_prob:1.0})
            money_predict[money_predict < 0] = 0

            a = result_2.copy()
            a[:, 1] = np.round(day_predict).reshape(-1)
            a[:, 2] = money_predict.reshape(-1)
            a[:,1] = np.where(a[:, 1] > 64, 64, a[:, 1])
            a[:,1] = np.where(a[:, 1] < 0, 0, a[:, 1])
            
            path = './temp/test2_predict_test2' + str(epoch_i) + '.csv'
            np.savetxt(path, a, delimiter=',', header='acc_id,survival_time,amount_spent', comments='')

        