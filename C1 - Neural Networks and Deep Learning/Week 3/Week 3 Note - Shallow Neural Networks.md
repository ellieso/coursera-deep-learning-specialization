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
- 학습 데이터가 입력되는 층인 "입력층(Input Layer)"
- 입력층과 출력층 사이의 층을 의미하는 "은닉층(Hidden Layer)"
- 결과(예측치)의 값이 산출되는 "출력층(Output Layer)"

신경망은 위의 3가지 층으로 구성되는데
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/92fbc64f-9e44-420c-b710-6229a3246574)

이러한 형태의 Neural Network를 2-layer NN이라고 합니다. Input Layer는 고정된 값이므로 포함시키지 않아 2-layer NNB이라고 부릅니다.
그래서 첫번째 layer는 Hidden Layer가 되고 두번째 layer는 Output Layer가 됩니다.

위의 Neural Network에서 순전파와 역전파를 진행하면 아래와 같습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/c40860e0-017e-4dd8-a9cd-357f4653ff85)

여기서 위 첨자로 [1]이라고 붙었는데 해당 값이 첫 번째 layer에 해당하는 값이라는 것을 의미합니다.
따라서, 첫번째 Layer에서 순전파를 진행하면서 z[1] = W[1]x + b[1] 와 a[1] = σ(z[1])를 구합니다. 첫번째 Layer에서 구한 a[1]을 사용해서 z[2]를 구하고 마지막 Layer이기 때문에 a[2]가 y-hat(예측치)가 됩니다. 이 결과값으로 Cost Function을 구하고 다시 역전파를 진행하면서 da[2], dz[2], dW[2], db[2]를 구하고 같은 방식으로 da[1],dz[1],dW[1],db[1]을 구합니다.

