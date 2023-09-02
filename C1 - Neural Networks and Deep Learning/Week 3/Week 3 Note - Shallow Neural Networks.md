Build a neural network with one hidden layer, using forward propagation and backpropagation.

## 학습목표

- Describe hidden units and hidden layers
- Use units with a non-linear activation function, such as tanh
- Implement forward and backward propagation
- Apply random initialization to your neural network
- Increase fluency in Deep Learning notations and Neural Network Representations
- Implement a 2-class classification neural network with a single hidden layer
- Compute the cross entropy loss

### Neural Networks Overview
신경망을 구현하기 앞서, 신경망(Neural Networks)의 전체적인 개요부터 알아보겠습니다.
저번 시간에 보았던 Logistic Regression에 기반해서 살펴보면, 
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/43f815f6-99dc-4e56-9e44-b36d81825bae)

위와 같은 Logistic Regression 모델에서 순전파(Forward Propagation)을 통해서 아래와 같이 식들을 도출했습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/856cb25c-c1a9-43ba-aca5-bc4b891997af)

그 결과 Cost Function인 L(a,y)를 구할 수 있었습니다. 그리고 역전파(BackPropagation)을 통해서 미분항 da와 dz를 구했습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/82c8ba5f-fec7-46e2-8dfb-49a17a5804b2)

다음으로 feature가 3개고 한 개의 hidden layer를 가진 Neural Network를 살펴보겠습니다.

신경망은 입력층, 은닉층, 출력층으로 구성됩니다.
- 학습 데이터가 입력되는 층인 "입력층(input layer)"
- 입력층과 출력층 사이의 층을 의미하는 "은닉층(hidden layer)"
- 결과(예측치)의 값이 산출되는 "출력층(output layer)"

신경망은 위의 3가지 층으로 구성되는데
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/92fbc64f-9e44-420c-b710-6229a3246574)

이러한 형태의 Neural Network를 2-layer NN이라고 합니다. input layer는 고정된 값이므로 포함시키지 않아 2-layer NN이라고 부릅니다.
그래서 첫번째 layer는 hidden layer가 되고 두번째 layer는 output layer가 됩니다.

위의 Neural Network에서 순전파와 역전파를 진행하면 아래와 같습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/c40860e0-017e-4dd8-a9cd-357f4653ff85)

여기서 위 첨자로 [1]이라고 붙었는데 해당 값이 첫 번째 layer에 해당하는 값이라는 것을 의미합니다.
따라서, 첫번째 Layer에서 순전파를 진행하면서 z[1] = W[1]x + b[1] 와 a[1] = σ(z[1])를 구합니다. 첫번째 Layer에서 구한 a[1]을 사용해서 z[2]를 구하고 마지막 Layer이기 때문에 a[2]가 y-hat(예측치)가 됩니다. 이 결과값으로 Cost Function을 구하고 다시 역전파를 진행하면서 da[2], dz[2], dW[2], db[2]를 구하고 같은 방식으로 da[1],dz[1],dW[1],db[1]을 구합니다.

### Neural Network Representation
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/f4fad2ea-ffc1-4a1f-b453-04be943906cc)

input layer/hidden layer/output layer, 3개의 층으로 이루어진 NN입니다. 하지만 보편적으로 input layer는 포함시키지 않기 때문에 2-layer NN이라고 합니다.(공식적인 표기, 논문에서 이렇게 쓰입니다.)

input layer에서 입력을 x라고 표현했는데 input feature값들은 a[0]으로 표기할 수도 있습니다. 여기서 a는 activation을 뜻하고, 이 activation은 다음으로 이어지는 layer로 전달하는 값입니다.
다음으로 hidden layer에서도 activation을 계산하는데, 이것은 a[1]으로 표현됩니다. 여기서 activation 노드가 총 4개있는데, 위에서부터 a1[1], a2[1], a3[1], a4[1]로 표현됩니다.
마지막으로 output layer에서 a[2]값이 계산되며, 이 값은 실수입니다.

### Computing a Neural Network's Output
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/44745bab-f400-4062-ab38-2e00256a8243)

Logistic Regression에서 한 개의 training example에 대해 위와 같이 계산되는 것을 볼 수 있습니다. z를 계산하고 sigmoid function을 적용해서 결과값이 나오게 됩니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/afb41b04-ef48-424d-a41f-3c42e62cf67f)

NN에서는 한 개의 training example에 대해서 다음과 같이 계산됩니다. Hidden Layer의 첫번째와 두번째 activation노드의 계산을 살펴보면, 첫번째 activation노드에서 z[1] = w1[1]Tx + b1[1]가 계산되고, 여기에 sigmoid function이 적용되어서 a1[1] = σ(z1[1])가 구해지게 됩니다. 마찬가지로 2,3,4 노드도 구하면 다음과 같이 구할 수 있습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/7345b062-9dc1-4b23-852d-db95c03062e9)

4개의 루프를 통해서 값을 구하는 건 비효율적이므로 벡터화를 시켜 값을 구할 수 있습니다.
먼저, z[1]을 벡터로 계산하는 방법을 살펴보면 W들을 가지고 행렬에 쌓아보면,
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/d9209438-8dec-4e5e-8f42-451c7d74dec6)

이런식으 4x3의 행렬을 만들수 있고 x도 같은 방법으로 

![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/38f251b6-1752-402e-8c75-7b897dc6cbfe)

이렇게 표현할 수 있다. 최종적으로 z[1]을 구현하게 되면,
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/2d4d2d9a-88b2-4443-9434-b739c0e9bb6a)

이러한 형태로 벡터화를 할 수 있고 구해진 z를 기반으로 a[1]도 구할 수 있습니다.

### Vectorizing Across Multiple Examples
이제까지 한 개의 training example에 대해서 NN에서의 순전파 진행를 알아봤고, m개의 training examples에 대한 순전파 진행을 알아보도록 하겠습니다. m개의 training exmples에 대해서 모든 activation노드를 구해야하고, 우리는 한 개의 training exmple에 했던 방식으로 m개의 example에 적용해야합니다.

![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/91df0400-1e39-45ca-b22b-04ab67eb84c7)

m개의 example에 대해서 계산하려면 위처럼 for문을 통해서 계산을 해야하지만, Vectorization을 하면 반복문없이 계산할 수 있습니다.

### Explanation for Vectorized Implementation
벡터화를 하는 이유는 프로그래밍 과정에서 비효율적인 계산을 하는 for 구문을 사용하지 않고 알고리즘 성능을 향상시키기 위해서 입니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/26034d09-bac4-4d45-980a-2a9e4537e6da)

위의 그림에서 W[1] * X 행렬을 보면, 행렬의 열은 입력데이터 x의 갯수를 나타내고 행렬의 행은 입력데이터 x의 차원수 m을 나타냅니다. 저기에 b[1]만 더해주면 z[1] 벡터가 만들어지는 것입니다.

### Activation Functions
지금까지 우리는 Sigmoid함수를 Activation Function으로 사용했는데, 때때로 다른 함수를 사용하는게 훨씬 나을 때도 있습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/f10a8805-b401-478a-82ff-754344c6025a)

앞서 배운 내용에서 Activation Unit을 구하는 두 단계가 있었고, 위와 같은 Sigmoid 함수를 사용했었습니다. 이 Sigmoid 함수 대신 다른 함수를 사용할 수도 있으니, 조금 더 일반화시켜서 σ가 아닌 g(z)를 사용해서 식을 표현해보면 다음과 같습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/c80e1bcf-d461-4ec9-9f7f-48ffb0cc498c)

어떤 경우에 따라 g(z)는 비선형 함수이고 Sigmoid가 아닐 수도 있습니다. Activation function 중에 Sigmoid 함수보다 항상 더 좋은 성능을 발휘하는 함수가 있는데, 바로 tanh 함수입니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/fcb78ef4-bb8b-4871-b5e2-e09812c126cd)

위와 같은 그래프를 가지며, -1에서 1 사이의 값을 가집니다. 또한 tanh 함수는 데이터를 중심에 위치시키려고 할 때, 평균이 0인 데이터를 갖게됩니다. 데이터를 중심에 위치시키는 효과 덕분에 다음 layer에서 학습이 더욱 쉽게 이루어지게 됩니다. 그래서 우리가 작업할 시 Hidden Layer에서는 Simoid는 거의 사용하지 않을 것이고, tanh 함수는 거의 항상 Sigmoid보다 우수하기 때문에 더 많이 사용될 것입니다.
한 가지 예외가 있는데, Output layer에서는 Sigmoid를 사용합니다. 왜냐하면 결과값 y는 0이거나 1인데, 예측값인 y_hat은 0과 1사이의 값이며 -1과 1사이의 값이 아니기 때문입니다. 그래서 이진분류 문제일 때는 Output layer에서 Sigmoid 함수를 사용할 수 있습니다.

Sigmoid와 tanh 함수는 한 가지 단점이 존재하는데, z가 매우 크거나 매우 작을 때, Gradient, 즉, 함수의 미분값(기울기)이 매우 작은 값이 됩니다. 따라서 z가 매우 크거나 작게 되면, 함수의 기울기는 거의 0이되고, Gradient Descent 알고리즘을 느리게 만드는 원인이 됩니다.
이러한 문제 때문에 머신러닝에서 자주 사용되는 activation 함수 중에 다른 하나는 ReLU(Rectified Linear Unit) 함수입니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/a5eba16d-4fe4-4a07-9e75-8938c86cb4bd)

ReLU 함수에서 z가 양수일 경우에는 미분값은 1이고, z가 음수이면 미분값은 0입니다. 수학적으로 z가 0인 지점, 즉 0에 매우 가까워지는 지점의 기울기는 구할 수가 없지만, 실제로는 걱정할 필요는 없습니다. 그래서 z가 0일 때, 우리는 미분값이 0이나 1인 것처럼 취급할 수 있고, 실제로 잘 동작합니다. 수학적으로는 사실 미분이 가능하지 않습니다. 그래서 우리는 출력값이 0 또는 1인 경우에, output layer에는 sigmoid 함수를 activation 함수로 선택할 수 있고, 나머지 다른 layer에서는 ReLU 함수를 선택할 수 있습니다. 만약 hidden layer에서 어떤 activation 함수를 사용해야될 지 모르겠다면 그냥 ReLU 함수를 사용하면 됩니다. ReLU의 한 가지 단점은 z가 음수일 때 미분값이 0이라는 것입니다.

여기에 약간 다른 버전의 ReLU가 있는데 바로 Leaky ReLU라 부르는 함수입니다. 이 함수는 z가 양수일 때는 ReLU와 동일하고 z가 음수일 때는 약간의 기울기를 갖게 됩니다. 
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/a4708874-c1e8-47cd-ac41-29350c7f04e8)

Leaky ReLU는 대게 ReLU보다 더 잘 동작하지만, 실제로 잘 사용하지는 않습니다.

ReLU와 Leaky ReLU의 장점은 이 함수에서 미분값 즉, 기울기가 0과는 다른 값이라는 점입니다. 실제로 ReLU 함수를 사용하면 NN이 종종 tanh나 sigmoid보다 훨씬 빠르게 동작하게 됩니다. 이러한 이유는 기울기에 의한 영향이 적다는 것인데 tanh나 sigmoid에서는 기울기가 0에 가까워져서 학습을 느리게 하는 경우가 있기 때문입니다. z가 음수일 때 ReLU함수의 기울기는 0인데, 실제로 hidden layer에서 z의 값은 0보다 충분히 클 것입니다. 그래서 학습이 빠르게 이루어질 수 있습니다.

Leaky ReLU에서 0.01이라는 파라미터를 곱해서 사용했는데 다른 값을 사용할 수도 있습니다. 물론 더 잘 동작하는 경우도 있겠지만 일반적으로는 다른 값을 쓰는 경우는 잘 없다고 합니다. 만약 다른 값을 사용하고 싶다면 시도해보고 얼마나 잘 동작하는지 살펴보고 좋은 결과를 갖게 된다면 사용해도 괜찮을 거 같습니다.

### Why do you need Non-Linear Activation Functions?
NN에서 왜 이런 비선형 activation 함수가 필요할까요? 
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/f206d4ff-ec7b-45d6-8d88-8ca3f6eafeab)

앞서 배운 공식에서 g(z) = z로 선형활성화 함수를 적용해보면,
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/e85eb01b-4dcb-4969-945e-4a96781abe2c)

이러한 공식을 도출할 수 있습니다. 이렇게 되면 이 모델은 y와 y_hat을 선형함수로 계산하게 되고, 이런경우에는 hidden layer가 무의미해집니다. 2개의 선형함수의 구성요소는 그 자체가 이미 선형함수입니다. 그러므로 비선형한 특성을 사용하지 않는 이상 더 흥미로운 결과를 낼 수 없을 것입니다.
선형함수를 사용하는 한 가지 경우가 있는데 그 경우는 g(z) = z인 경우이다. 앞서 나온 집값 예측 문제 와 같이 y의 값이 실수로 나온다면 사용이 가능할 것이다. 이 경우에도 hidden layer에는 선형함수를 사용하지 않고 output layer에서만 사용됩니다.

결론적으로 선형함수를 활성화 함수로 사용하게 되면 신경망을 거치는 의미가 사라지고 경사하강법을 사용하지 못하게 됩니다.

### Derivatives of Activation Functions
역전파를 진행할 때 우리는 activation 함수의 미분, 기울기를 구해야 합니다. 각 함수의 미분을 하는 방법을 알아보겠습니다.
먼저 시그모이드 함수입니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/e91fd55d-a95f-4bca-8268-ac998fcd0ef1)

만약 g(z)가 시그모이드 함수라 함수의 기울기는 d/dz*g(z)이며 미적분을 통해 z에서 g(x)의 기울기라는 것을 알게됩니다. 결과적으로 sigmoid 함수의 미분값과 같아지는 것입니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/7ebe94c0-3f47-44d2-bbd1-6d6260d0db9d)

그 다음 tahn함수입니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/e25bd4bd-e169-4862-a0b0-8504100c12c6)
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/93016d76-54e7-48a4-b2eb-2e1161d00d28)

다음은 ReLU함수입니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/2212ee82-17cf-4b3d-ba60-006ca017e892)

여기서 실제 수학적으로는 z가 0일 때는 미분값을 정의할 수 없지만 소프트웨어에 도입시키는 경우 0 또는 1로 지정합다. z가 정확히 0이 될 확률은 너무나도 작은 확률이라서 별로 상관이 없다고 생각하면 됩니다. 즉, z가 정확히 0일 때, 어떤 미분값을 설정하는지는 별로 상관이 없습니다.

다음은 Leaky ReLU함수입니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/e1b536cf-4155-4e55-bb67-dd78a2908185)

ReLU함수와 동일하게 z가 정확히 0일 때는 기울기는 정의되지 않지만, 0.01 또는 1로 설정해도 문제가 되지 않습니다.

### Gradient Descent for Neural Networks
이번에는 1-hidden layer NN에 대해서 Gradient Descent를 진행해보겠습니다.

필요파라미터는 W[1], b[1], W[2], b[2] 이고, 각 layer의 차원을 계산하기 위해 각 layer의 노드 개수를 nx = n[0], n[1], n[2] = 1로 정의합니다. 그러면 파라미터들이 이러한 차원의 형태를 갖습니다.
W[1] = (n[1], n[0]) / b[1] = (n[1], 1) / W[2] = (n[2], n[1]) / b[2] = (n[2], 1)

알고리즘의 매개변수를 훈련하기 위해서는 Gradient Descent를 행해야 합니다. 신경망을 훈련시키는 동안 모든 0이 아닌 값으로 랜덤하게 파라미터를 초기화하는 것이 중요합니다.
Gradient Descent는 다음과 같이 방식으로 진행됩니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/9b32e057-8939-404c-93a8-dbb6a9e6a39e)

공식에 대한 설명은 뒤의 강의에서 자세히 설명될 예정입니다.
위의 식을 순전파와 역전파의 공식으로 정리해보면 다음과 같습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/db9ab89b-54c9-416d-87b2-7d02a25d092e)

dZ[1]을 구하는 식에서 차원은 둘다 (n[1],m)이고, 곱은 행렬곱이 아닌 요소곱입니다. 파라미터로 axis = 1은 행은 유지하면서 각 행의 열을 모두 더하라는 의미이며 keepdims = True 파라미터를 통해서 계산된 결과가 rank 1 Matrix가 되지 않도록 차원을 유지시켜주는 역할을 합니다.

### Backpropagation Intuition
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/40b2bd79-7d0a-418f-bfa7-461343440541)

역전파는 순전파와 반대방향으로 진행됩니다. (da[2] -> dz[2] -> dw[2] -> db[2] -> da[1] -> dz[1] -> dw[1] -> db[1])
dz[2]는 a[2]-y와 같고 dw[2]는 dz[1] * a[1]T이며, db[2] = dz[2]입니다. 그리고 dz[1]은 w[2]T * dz[2] * g[1]'(z[1])과 같습니다. dw[1]은 dz[1] * a[0]T와 같고  db[1]은 dz[1]과 같습니다.
값을 정리하면 다음과 같습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/e2179f4f-507d-4028-b872-4e058b164dcc)


### Random Initialization
NN에서 Weight(파라미터 W)를 랜덤으로 초기화하는 것은 중요합니다. 이전 Logistic Regression에서는 weight를 0으로 초기화하는 것이 괜찮았지만, NN에서 모든 weight를 0으로 초기화하고 Gradient Descent를 적용하면 제대로 동작하지 않습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/84e3448c-e8e1-4d22-8bae-a00854bc7c1b)

예를 들어 W[1], b[1], W[2], b[2]를 0으로 초기화했다고 할 때, b는 0으로 설정해도 문제가 없지만 W를 0으로 초기화하는 것은 문제가 됩니다. 그 이유는 a1[1]과 a2[1]이 동일할 것이고 역전파를 계산할 때 dz1[1]과 dz2[1]도 동일해지게 됩니다.
그래서 w를 0으로 초기화하면 W[1] hidden unit 모두 똑같은 함수로 계산되고, 2개의 hidden unit이 결과값에 동일한 영향을 주기 때문에 iteration 후에도 동일한 상황이 됩니다. 두 개의 hidden unit이 계속해서 대칭(Symmetric)이 됩니다. 그렇게 되면 hidden unit이 두 개 이상이 있는게 무의미해집니다.

그래서 이 문제를 해결하기 위해서는 매개변수를 무작위로 초기화해야 합니다. W[1] 값을 np.random.randn((2.2)) * 0.01으로 설정해 무작위성을 주고, b[1]는 대칭성문제가 없으므로 np.zero((2,1))로 설정해도 문제가 없습니다.
W[1]값을 설정할 때 0.01을 곱해준 이유는 가중치는 매우 작은 무작위 값으로 초기화하는 것을 선호하기 때문입니다. 가중치가 너무 크면 활성화 값을 계산할 때, 학습속도가 느려질 수 있습니다. 꼭 0.01이 아니더라도 더 작은 값으로 입력해도 문제는 없습니다.
