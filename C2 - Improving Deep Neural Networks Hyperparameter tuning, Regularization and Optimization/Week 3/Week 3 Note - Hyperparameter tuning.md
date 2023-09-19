Explore TensorFlow, a deep learning framework that allows you to build neural networks quickly and easily, then train a neural network on a TensorFlow dataset.

## 학습목표

- Master the process of hyperparameter tuning
- Describe softmax classification for multiple classes
- Apply batch normalization to make your neural network more robust
- Build a neural network in TensorFlow and train it on a TensorFlow dataset
- Describe the purpose and operation of GradientTape
- Use tf.Variable to modify the state of a variable
- Apply TensorFlow decorators to speed up code
- Explain the difference between a variable and a constant

### Tuning process
신경망을 변경하려면 다양한 하이퍼파라미터를 설정해야하는 것을 배웠습니다. 
 그렇다면 최적의 하이퍼파라미터 세팅을 찾는 방법을 알아보도록 하겠습니다.
Deep Neural Network를 훈련하는 과정에서 힘든 부분 중 하나는 많은 양의 하이퍼 파라미터 개수입니다.
이런 하이퍼 파라미터들은 학습률을 나타내는 α값 모멘텀을 쓰는 경우 모멘텀 값, 아니면 Adam 최적화 알고리즘의 하이퍼파라미터인 β1, β2, ϵ이거나 layer의 수를 선택해야할 수도 있습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/a5de6b9e-5c6a-4222-b221-6155e3c6ffd2)

하이퍼파라미터 중에서 α가 가장 중요하며, 다음으로 노란색 박스의 하이퍼파라미터(β, # of hidden units, mini-batch size)가 중요하고, 다음으로는 보라색 박스인 # of layer, learning rate decay가 중요하다고 할 수 있습니다.

하이퍼파라미터 튜닝을 할 때 어떻게 설정해서 탐색해야 할까요?
머신러닝 알고리즘 초기에는 2개의 하이퍼파라미터가 있는 경우에 점들을 격자판 형식으로 샘플링하는 경우가 많았습니다. 이러한 방법은 하이퍼 파라미터의 개수가
비교적 많지 않은 경우에 잘 동작하는 편입니다.
딥러닝의 경우에 추천하는 방법은 임의로 설정하는 것 입니다. 이러한 방법을 사용하는 이유는 어떤 하이퍼파라미터가 가장 중요할 지 미리 알기가 어렵기 때문입니다.
예를 들어서, 하이퍼파라미터1은 learning rate인 α이고 하이퍼파라미터2는 Adam 알고리즘의 ϵ이라고 해보겠습니다. 이런 경우에는 α는 매우 중요하고 ϵ은 거의 중요하지 않을 것입니다.  따라서, 왼쪽같이 격자판 형식으로 샘플링을 진행하면 25개의 모델을 학습했지만, 5개의 α값으로만 시도할 수 있을 것입니다.  반대로 샘플링을 무작위로 진행한다면, 각각 25개의 α를 시도하게 됩니다.
어떤 하이퍼파라미터가 더 중요한지 미리 알아내기 어렵기 때문에, 격자판 형식으로 샘플링하는 것보다 무작위로 샘플링하는 방법이 더 많은 하이퍼파라미터 값을 탐색해볼 수 있기 때문에, 더 광범위하게 학습을 진행할 수 있습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/784066e6-935e-4bb3-b61c-675facab9de0)

이런 이차원 격차에서 이 포인트가 가장 잘 작동한다고 해보겠습니다. 하이퍼파라미터의 더 작은 영역을 확대한 다음 이 공간 내에서 더 많은 밀도를 샘플링 할 수 있습니다. 그런 다음 더 작은 정사각형으로 조밀하게 샘플링 할 수 있습니다. 하이퍼파라미터의 다양한 값을 시험해봄으로써 훈련 세트 목표에서 가장 잘 할 수 있는 값이나 개발 세트에서 가장 잘하는 값 또는 하이퍼파라미터 검색 프로세스에서 최적화하려는 모든 값을 선택할 수 있습니다. 

### Using an Appropriate Scale to pick Hyperparameters
여러범위의 하이퍼파라미터에서 임이의 샘플링을 통해 더 효율적으로 하이퍼파라미터 공간을 검색하는 방법을 배웠습니다. 임의로 샘플링을 하는 것은 균일화된 임의의 작업을 뜻하는 것이 아닙니다. 대신에 적절한 척도를 선택하여 하이퍼파라미터를 탐색해야 합니다.
은닉단위의 수인 n[l]을 찾으려고 한다고 가정하면 좋은 범위는 50에서 100 사이라고 해보겠습니다. 이 숫자 라인 내에서 임의의 숫자 값을 선택할 수 있습니다. 또는 신경망 네트워크 층의 개수를 결정하려고 한다고 하면 총 층의 개수가 2개에서 4개 사이여야 한다고 생각할 수도 있습니다. 그러면 2,3,4,에  따라 랜덤으로 균일하게 추출하는 것이 합리적일 수 있습니다.
이러한 몇개의 예제들과 같이 균일화된 임의의 방법으로 샘플링 하는 경우 생각하고 있는 범위가 합리적인 방법일 수도 있습니다. 하지만 이런 방법이 모든 하이퍼 파라미터에서 적용되는 것은 아닙니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/3bfd20e7-a530-4489-a0c6-87565fc87b71)

학습률 알파값을 검색한다고 해보겠습니다. 그리고 이 경우에 0.0001이 낮은 경계선이고 또는 높은 경계선이 1일 수 있겠습니다. 약 90%의 값이 0.1과 1 사이에 있을 것입니다. 즉, 샘플링의 90% 정도를 0.1에서 1 사이의 값에 집중하게 되는 것이고, 10% 정도만이 0.0001과 0.1 사이의 값에 집중하는 것입니다.
그러므로 로그 스케일해서 하이퍼파라미터를 검색하는 것이 더 합리적으로 보입니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/540b51b5-aa38-44ba-a93e-bf50e14e59e8)

지수가중평균을 계산하는데 사용되는 하이퍼파라미터 β 샘플링은 조금 헷갈릴 수 있습니다. 적절한 β의 값 0.9에서 0.999사이의 값이라고 하면, 우리가 탐색해야되는 범위도 0.9 ~ 0.999가 됩니다. 위와 같은 범위에서도 로그 스케일로 탐색해야 합니다. 가장 쉬운 방법은 1-β의 범위를 찾는 것이고 0.0001~0.1사이의 값을 찾는 것입니다.
지수가중평균에서 β값은 1에 가까워질수록 결과에 민감해집니다. 0.9000에서 0.9005로 0.0005가 증가하면 큰 영향을 미치지 않을 것입니다. 하지만 0.9990에서 0.9995로 동일하게 0.0005가 증가하면, 결과에 많은 차이를 보이게 될것입니다.

### Hyperparameters tuning in practice: Pandas vs. Caviar
하이퍼파라미터 탐색을 위한 팁과 기술은 다음과 같습니다. 딥러닝은 오늘날 많은 분야에서 적용되어 있는데 한 분야에서 사용되는 적절한 하이퍼파라미터가 다른 분야에 적용될 수도 있고 아닐수도 있습니다.
모델을 학습하고 적절한 하이퍼파라미터의 값을 찾았더라도, 시간이 지나면서 데이터가 변하거나, 데이터센터의 서버를 업그레이드하거나, 이러한 변화들 때문에 우리가 찾은 하이퍼파라미터의 값이 더 이상 최적의 파라미터가 아니게 될 수 있습니다. 그러므로 다시 테스트하거나 하이퍼파라미터를 재평가하는 것이 필요합니다.
하이퍼파라미터를 탐색하는 방법은 2가지가 있습니다. 
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/22ccd0e5-391a-4411-a34e-14575c6391f3)

첫번째로는 데이터의 양이 매우 많고, 연산을 위한 Resource가 많이 없을 때 적용되며, Resource는 CPU나 GPU를 의미하고, 충분하지 않아서 한 번에 1가지 모델만 학습할 수 있을 경우를 의미합니다. 예를들어 day 0일때, 파라미터를 임의로 초기화시키고 학습을 합니다. 그리고 learning curve(cost J에 대한 그래프)를 보면서, 적절하게 동작한다면 감소하는 learning curve를 볼 수 있을 것입니다. 그렇게 day 1이 지나면서 cost가 감소하는 것을 확인하고 learning speed를 약간 높힐 수 있고, day 2의 결과를 확인합니다. day 2의 성능도 좋다는 것을 확인할 수도 있는데, 이런 방식으로 진행하다가 보면 어느날에는 learning speed가 너무 컸다는 것을 깨달을 수 있는데, 이렇게 되면 이전 모델로 다시 돌아갈 것이입니다. 이 방법은 성능을 관찰하고 인내심 있게 학습률을 높이거나 낮추는 것입니다. 이 경우는 일반적으로 많은 모델을 동시에 훈련할 수 없는 경우에 발생합니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/37e198d1-d597-4fdb-803a-dac2886bc8d2)

다른 방법으로는 병렬방식으로 여러모델을 훈련시키는 방법입니다. 하이퍼파라미터 일부를 설정할 수 있는데 혼자 작동하게 놔두는 것입니다.
Babysitting one model방법은 판다 접근 법이라고 부르고 Training many models in parallel 캐비어 접근법이라고 불립니다. 이 두개의 접근방식 중 고르는 방법은 얼마나 산출자원을 가지고 있느냐에 따라 달라집니다. 
