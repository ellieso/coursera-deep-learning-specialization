Develop your deep learning toolbox by adding more advanced optimizations, random minibatching, and learning rate decay scheduling to speed up your models.

## 학습목표
- Apply optimization methods such as (Stochastic) Gradient Descent, Momentum, RMSProp and Adam
- Use random minibatches to accelerate convergence and improve optimization
- Describe the benefits of learning rate decay and apply it to your optimization

### Mini-batch Gradient Descent
신경망 알고리즘을 더 빨리 훈련시킬 수 있는 최적화 알고리즘에 대해 배우겠습니다. 딥러닝을 빅데이터에 훈련시키면 그 속도가 매우 느릴 것입니다. 이럴 경우 빠른 최적화 알고리즘과 양질의 최적화 알고리즘을 갖는 것이 효율성의 속도를 높일 수 있습니다.

일반적인 Gradient Descent는 Batch Gradient Descent라고 부르며, 파라미터를 업데이트할 때 모든 training examples m에 대해서 gradient를 계산한 다음에 파라미터를 업데이트하게 됩니다. 즉, GD를 진행하기 전에 전처리 과정(Gradient 계산)이 필요하고, m이 매우 커질수록 느려지게 됩니다.

전체를 아주 작은 훈련세트로 나눠서 미니 배치들이라고 부르겠습니다. 만약 5,000,000개의 training 샘플이 있고 이 값을 1000개로 미니배치하면 x1~x1000를 x{1} / x1001~x2000을 x{2} ...으로 표현하게 됩니다. 그렇게되면 총 5000개의 미니배치가 생성되게 됩니다. 마찬가지로 y에 대해서도 동일작업을 수행합니다. 미니배치는 X{t}, Y{t}형태의 index를 가지게 됩니다.

미니배치의 진행방식은 다음과 같습니다. trainin set에서 미니배치경사하강볍을 실행하려면 각각 미니배치마다 실행해주면 됩니다.
1. 입력값에 대한 순전파를 도입합니다.(Z[l] = W[l]X{t} +b[l])
2. 활성함수에 적용합니다. A[l] = g[l](Z[l])
3. A[L] = g[L](Z[L])이 될 때까지 시도합니다.
4. 비용함수를 산출합니다.
5. 역전파를 도입합니다. (W[l] = W[l] - αdW[l], b[l] = b[l] - αdb[l])
여기서 한번의 for문 진행을 '1 epoch'라고 하며, 여기서는 한 번의 Gradient Descent에서 총 5000 epoch를 진행합니다. 따라서 전체 iteration을 위한 또 하나의 for loop가 필요합니다.
