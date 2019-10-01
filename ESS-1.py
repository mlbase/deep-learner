
import numpy as np
import matplotlib.pyplot as plt




class ESS:
    pi=[0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.118,0.14,0.14,0.118,0.14,0.14,0.14,0.14,0.118,0.118,0.118,0.118,0.118,0.118,0.06,0.06] #가격표
    delta_ESS = 50      ##delta는 충방전 기준값

    def __init__(self, Capacity=int):
        self.capacity =Capacity

    def reset(self):
        self.SOE_ini = self.capacity * 0.5
        self.SOE_min = self.capacity * 0.1
        self.SOE_max = self.capacity * 0.9
        self.state = self.SOE_ini
        self.time = 0
        self.zeta =1000     ##가격 패널티

        self.state_zip = []
        self.action_zip = []
        self.reward_zip = []

    def make_action(self, num_action = int):   ##action은 경우의 수인데 충방전 용량에 대한 경우 ex)20, 40,60 ,...
        self.action_choice = []
        for i in range(int(-num_action/2),int(num_action/2)):    ## 충방전 경우의 수
            self.action_choice.append(self.delta_ESS*i)         ## 충방전 전력의 경우에 대한 행렬 축적

    def act(self):
        self.action = np.random.choice(self.action_choice)      ## 위의 충방전량에 대한 무작위 추출,  인위적인 경우도 추가할 수 있도록 수정요망
        self.state = self.state + self.action
        self.time += 1
        self.calculation_reward()
        self.state_zip.append(self.state)
        self.action_zip.append(self.action)
        self.reward_zip.append(self.reward)  ## reward가 축적

        if self.time == 24:
            self.Total_reward = sum(self.reward_zip)

    def calculation_reward(self):
        if self.state > self.SOE_max:
            self.reward = -self.pi[self.time - 1] * self.action - self.zeta * self.action
        elif self.state < self.SOE_min:
            self.reward = -self.pi[self.time - 1] * self.action + self.zeta * self.action
        else:
            self.reward = -self.pi[self.time - 1] * self.action


ESS=ESS(Capacity=3000)



# Initialize table with all zeros
Q = np.ones([24,21])*(-20000)

# Set learning parameters
learning_rate = .85
dis = .99
num_episodes = 2000


# create lists to contain total rewards and steps per episode
rList = []
ESS.make_action(num_action=21)

for i in range(num_episodes):
    # Reset environment and get first new observation
    ESS.reset()

    state = ESS.time
    rAll = 0


    # The Q-Table learning algorithm
    for j in range(0,24):
        action = int(np.argmax((Q[state, :] + np.random.randn(21,) / (i + 1))))

        # Get new state and reward from environment
        new_state ,ESS.action_choice,  __ = Q.step(action)

        # Update Q-Table with new knowledge using learning rate
        Q[state, action] = ESS.calculation_reward() + dis * np.max(Q[new_state, :])
        state = new_state
        rAll += ESS.calculation_reward()


    rList.append(rAll)

print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
#plt.bar(range(len(rList)), rList, color='b', alpha=0.4)
plt.show()