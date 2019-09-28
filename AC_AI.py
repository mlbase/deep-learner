
import numpy as np
import tensorflow as tf

class Temperature:
    learning_rate = 0.005
    training_rate = 1500
    batch_size = 500
    T_set=24
    T_out = [23, 23, 23, 23, 24, 24, 27, 28, 28, 31, 31, 32, 33, 33, 32, 32, 30, 30, 29, 27, 27, 26, 25, 24]
    
#make session
    def __init__(self,sess,name):
        self.name=name
        self.build_network()
        self.session=sess


    def build_network(self):
        with tf.variable_scope(self.name): 
            self.keep_prop=tf.placeholder(tf.float32) #drob out 을 설정하는 추가 변수
            self.x=tf.placeholder(tf.float32,[None,4]) #텐서(placeholder)
            self.y=tf.placeholder(tf.float32,[None,1])
#placeholder : 학습예제 variable : 실제 학습을 하는 변수
            self.W1=tf.Variable(tf.random_normal([4,40]),name='weight1') # tf.variable 값이 정해지지 않다. weight wx+b , random 4x40 행렬 
            self.b1=tf.Variable(tf.random_normal([40]),name='bias1') # bias wx+b
            self.L1=tf.nn.relu(tf.matmul(self.x,self.W1)+self.b1) # 음수일때는 0으로 나오게 양수일때는 그대로 받게하는 함수,matmul(matrix multiple)
#신경망
            self.W2=tf.Variable(tf.random_normal([40,1]),name='weight1')
            self.b2=tf.Variable(tf.random_normal([1]),name='bias2')
# y=wx+b 행렬곱을 통해 신경망을 형성
            self.hypothesis=tf.matmul(self.L1,self.W2)+self.b2 #wx+b step2

            self.cost=tf.reduce_mean(tf.square(self.hypothesis-self.y)) # redcuce_mean 표준편차를 최소, self.y와 딥러닝한 결과값을 뺄셈해서 제곱하고 평균구하기 

            self.optimizer=tf.train.AdamOptimizer(Learning_rate=self.learning_rate).minimize(self.cost) # adamoptimizer라는 알고리즘을 씀

    def train(self, input_data, output_data, keep_prop=0.7): #keep_prop : drob out 을 설정하는 추가 변수
        return self.sess.run([self.cost, self.optimizer],feed_dict={self.x:input_data, self.y:output_data, self.keep_prop:keep_prop})
      
# feed_dictionary 는 placeholder로 변수선안한 변수만 사용가능!, 피딩 : 세션을 실행할때마다 데이터를 저장
    def predict(self, test_data=HVAC.reward_cal(), keep_prop=1.0):
        return float((self.sess.run(self.hypothesis,feed_dict={self.X:[[test_data,self.T_set,T_out[HVAC.time-1],HVAC.temp]]})))
    
    def update_network(self):
        


class HVAC:
    pi = [0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.118, 0.14, 0.14, 0.118, 0.14, 0.14, 0.14, 0.14, 0.118,
          0.118, 0.118, 0.118, 0.118, 0.118, 0.06, 0.06]

    def __init__(self, T_min, T_max,T_ini,epsilon,delta_energy, num_energy_level):
        self.Tmin = T_min
        self.Tmax = T_max
        self.Tset = (self.Tmin+self.Tmax)/2
        self.Tini = T_ini
        self.epsilon = epsilon
        self.delta_HVAC = delta_energy
        self.num_energy_level = num_energy_level
        self.make_action(self.num_energy_level)

    def make_action(self, num_energy_level):
        self.action_choice=[]
        for i in range(0, num_energy_level+1):
            self.action_choice.append(i*self.delta_HVAC)

    def reset(self):
        self.time = 0
        self.action = np.random.choice(self.action_choice)
        self.state = self.action
        self.temp = self.Tini
        self.reward = None

        self.state_zip=[]
        self.action_zip=[]
        self.temp_zip=[]
        self.reward_zip=[]

    def act(self,action):
        self.move(action)
        self.state =  self.action
        self.temp = Temperature.predict(self.state)
        self.time+=1
        self.reward_cal()
        self.if_done(self.time)

        self.state_zip.append(self.state)
        self.action_zip.append(self.action)
        self.temp_zip.append(self.temp)
        self.reward_zip.append(self.reward)

    def if_done(self, time):
        if self.time == 24:
            self.done = True
        else:
            self.done = False


    def move(self,choice=False):
        self.action = self.delta_HVAC*choice
        return self.action

    def reward_cal(self):
        if self.temp > self.Tset:
            self.reward = -(self.pi[self.time-1]*self.state+self.epsilon*(self.temp-self.Tset))
        elif self.temp < self.Tset:
            self.reward = -(self.pi[self.time-1]*self.state+self.epsilon*(self.Tset-self.temp))
        else:
            self.reward = -(self.pi[self.time-1]*self.state)

HVAC = HVAC(T_min=22, T_max=25,T_ini=20,epsilon=50,delta_energy=20, num_energy_level= 20)
HVAC.reset()
HVAC.act(1)
print(HVAC.state)
