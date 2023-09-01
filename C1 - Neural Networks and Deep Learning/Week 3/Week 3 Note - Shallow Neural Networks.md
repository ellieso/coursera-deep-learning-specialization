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
