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

여기서 b[l]은 무시될 수 있는데, batch norm에서 평균값을 구하고, input에 평균값을 빼기 때문에 상수항을 더하는 것은 아무런 효과가 없어지기 때문입니다.
