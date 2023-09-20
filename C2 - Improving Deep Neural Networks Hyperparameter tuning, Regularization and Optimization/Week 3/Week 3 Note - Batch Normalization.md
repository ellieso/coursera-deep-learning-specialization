### Normalizing Activations in a Network
딥러닝이 상승하는 시점에서 가장 중요했던 아이디어 중 하나는 정규화라는 알고리즘 입니다. 배치 정규화를 통해 하이퍼파라미터 검색 문제가 훨씬 쉬워지고 신경망을 더 튼튼하게 만들어줍니다.
모델을 훈련하는 경우 입력기능을 정규화하면 평균을 계산할 때 학습 속도를 높일 수 있습니다. 더 깊은 모델의 경우 입력이 x만 있는 것이 아니라 activation도 있는데 이 경우에 Batch Normalization 알고리즘을 적용합니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/576236ce-d0af-4d6a-ae9e-3ad8b041c87f)

위의 신경망에서 output layer 이전 layer의 입력은 a[2]이고 이 값을 정규화할 수 있는지가 문제입니다. 그리고 이 방법은 w[3],b[3]을 더 빨리 학습시키는 방법이고 a[2]는 w[3],b[3]의 학습에 영향을 줍니다. 이 방법이 Batch Normalization입니다. 실제로는 a2가 아닌 z2의 값을 정규화하는 것입니다.
Batch Normalization을 수행하는 방법은 다음과 같습니다. 신경망에서 어떤 중간값이 있는데 z1에서 zm까지 은닉값이 있을 때 계산식은 다음과 같습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/cd44f041-5ed5-49a5-9d85-0179fc8bd9c6)

마지막의 γ와 β는 모델에서 학습하는 parameter이고, Gradient Descent/Momentum/RMSprop/Adam을 통해서 학습할 수 있습니다.  γ와 β를 사용하지 않으면 정규화를 통해서 hidden unit은 항상 평균이 0, 분산이 1이 되므로  γ와 β를 통해서 z틸의 평균이 원하는 값이 되도록 해줍니다.

### Fitting Batch Norm into a neural network
Batch Normalization이 어떻게 신경망에 적용되는지 알아보겠습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/8bfb70bb-0b0f-4697-8a06-21fb0e8854f0)

이렇게 생긴 신경망이 있다고 할 때, 첫번째로 z의 값을 계산하고 다음으로 활성화함수를 적용하여 a를 구합니다. Batch Normalization을 적용하지 않는다면 첫번째 은닉층에 맞는 입력 x를 가질 수 있고 z1을 구한다음 a1의 값을 구합니다. 하지만 z1에서 Batch Normalization을 적용시키는데 이 값은 파라미터 γ와 β의 적용을 받을 것입니다. 이 값은 g[1](z틸더[1])이됩니다. z와 a산출과정에서 Batch Normalization이 일어나게 됩니다. (여기서의 β는 앞서 배웟던 adam이나 momentum의 값과는 다릅니다.) 새로운 파라미터인 γ와 β도 W,b와 같은 방법으로 학습되며, β[l] = β[l] - αdβ[l], γ[l] = γ[l] - αdγ[l]으로 계산됩니다.
mini-batch에 Batch Normalization을 적용하면 다음과 같이 적용됩니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/6a566bd3-693f-4ff2-a5d1-55df4714a723)

여기서 b[l]은 무시될 수 있는데, Batch Normalization에서 평균값을 구하고, input에 평균값을 빼기 때문에 상수항을 더하는 것은 아무런 효과가 없어지기 때문입니다.
Batch Normalization을 mini-batch에 사용해서 경사하강 진행은 아래와 같이 수행됩니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/f6d40b6f-58f7-4129-9071-c5a6f22484be)

### Why does Batch Norm work?
Batch Normalization이 작동하는 이유는 무엇일까요? 한가지 이유는 X라는 입력특성을 정규화하여 평균값을 0, 분산 1 로 만들었는데 이것이 학습의 속도를 증가시킵니다. 모든 특성을 정규화하기 때문에 input값이 비슷한 범위를 갖게 되고 학습 속도가 증가하는 것입니다. 그리고 이 정규화는 input layer 뿐만아니라 hidden unit에도 적용됩니다.
두번재 이유는 Batch Normalization을 통해서 모델이 w에 변화에 덜 민감해지기 때문입니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/23634857-84bf-4614-8322-69e84cab18d1)

위와 같은 Logistic Regression에서 우리가 모든 데이터를 검은 고양이의 사진으로 학습을 진행했다고 가정한다면, 이 모델이 오른쪽의 색깔이 있는 고양이에 적용하면, 분류기가 잘 동작하지 않을 수 있습니다. 이렇게 데이터의 분포가 변하는 것을 covariate shift라고 불리는데, 만약 입력 x의 분포도가 변경되면 학습 알고리즘을 다시 학습시켜야 할 수도 있습니다.

![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/a93e1a3d-cdc6-4214-bf4f-99702bf3b180)
이러한 신경망이 있다고 할 때, 3번째 레이어에서는 w[3]과 b[3]을 학습했을 것입니다. 그리고 3번째 layer를 기준으로 특정값들을 이전층에서 가지고 오는데 ground True Y값으로 만들기 위해서는 몇가지 작업을 진행해야 합니다.
그렇다면 세번째 layer 기준에서 살펴보면, a[2]는 항상변하는 것이고 언급했었던 covariate shift 문제가 발생하게 됩니다. 여기서 Batch Normalization의 역할이 이런 hidden unit의 분포가 변경되는 정도를 줄여줍니다. batch normalization은 이전 layer의 파라미터들을 업데이트 하면서 변하더라도평균값과 분산은 똑같게 유지해줍니다
여기서 γ와 β는 새로 학습해야 하는 파라미터 입니다. 결과적으로 Batch Normalization은 input 값이 변해서 생기는 문제들을 줄여줍니다. 입력의 분포도가 약간 변경될 수는 있지만, 변경되는 정도를 줄여주고 결과적으로 다음 layer에서 학습해야되는 양을 줄여주어서 더 안정적으로 학습할 수 있게 합니다.
추가적으로 Batch Normalization은 Regularization의 효과가 있습니다. 각 mini-batch에서는 해당되는 mini-batch의 데이터로만 mean/variance를 계산했기 때문에 전체 샘플로 했을 구했을 때보다 더 noisy하고, 결과적으로 z틸다 또한 noisy해질 것입니다. 그 결과, dropout과 유사하게 hidden layer activation에 noisy를 더하게 되는 것이고, 이로 인해서 regularization의 효과가 발생하는 것입니다.

### Batch Norm at test time
Batch Norm은 데이터를 한 번에 하나의 미니 배치로 처리하지 test time에는 예제를 한 번에 하나씩 처리해야 할 수도 있습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/465743d8-69d2-4908-a238-dded6270883a)

Batch Norm에 사용할 방정식입니다. 싱글 미니 배치 안에서 평균을 계산하기 위해 Z(i)값과 미니 배치를 합산합니다. 하지만 test time에서는 64,128,256개의 예시들을 한번에 처리할 수 있는 미니배치가 없을 수 있습니다. 그렇기 때문에 μ와 σ2을 구하기 위한 어떠한 또 다른 방법이 필요합니다. 그리고 만약 1개의 예시밖에 없는 경우 1개의 값을 가지고 평균과 편차를 구하는 것은 말이 안됩니다.
test time에서 신경망을 적용한다는 것 별도의 μ와 σ2의 예측값을 구하는 것입니다. 지수가중평균방식을 이용하여 그 평균은 미니배치에서의 값으로 하여 구합니다. 지수가중평균방식을 사용해서 학습에 사용되는 값들을 사용해서 예측값을 구하기 때문에 running average라도 불리기도하며, 꽤 정확하고 안정적인 편입니다.
