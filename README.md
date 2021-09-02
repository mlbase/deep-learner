# HEMS 졸업작품
## 설계과정
### 들어간 기술 STACK
1. python 3.6
2. OpenAi Q-learning
3. E-greedy algorithm
4. no-neural network
5. matlab
6. numpy package
---
### HEMS의 원리
#### Temparatre in 과 Temparature out 을 통한 온도 예측모델
-matlab을 통해서 T in 과 T out formula를 통해서 기존 온도 변화와 전력 변화 측정

-E-greedy 를 통한 학습(ESS 방전량이나 충전량 에어컨 발전량 등을 학습할 데이터로 설정)

-학습을 평가시 전력소모량과 실내온도를 바탕으로 평가
