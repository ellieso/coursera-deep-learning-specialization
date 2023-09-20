### Softmax Regression
이제까지 배웠던 분류분제에서는 0또는 1로 분류되는 이진분류를 사용했습니다. Softmax 회귀라는 로지스틱 회귀의 일반화가 있습니다. 단지 두 개의 클래스를 인식하는 것보다 하나 또는 여러 클래스 중 하나를 인식하려고 할 때 예측을 덜하게 됩니다. 
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/571c8772-e225-44c4-8639-7bc986ad417c)

이러한 형태의 값을 예측하려고 할 때, 고양이를 class 1, 강아지를 class 2, 병아리를 class 3이라고 하고, 3개의 클래스에도 해당이 되지 않으면 class 0이라고 하겠습니다.
대문자 C를 사용해서 클래스의 번호를 나타내겠습니다. 그렇다면 이 경우에는 C = 4입니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/6e40a480-6716-4f89-bd5e-d531cd8892b0)

이 output layer은 4개의 class들의 확률을 나타내는 것이고, 첫 번째 노드는 other class일 확률을 결과값으로 나타내고, 두 번째 unit은 고양이일 확률, 세번째 unit은 강아지일 확률, 네번째 unit은 병아리일 확률을 나타냅니다. 여기서 결과값은 4x1차원의 벡터입니다.이렇게 위와 같이 여러 클래스에 대해서 확률값을 나타낼 수 있도록 하는 layer(위와 같이 4개의 hidden unit을 갖는)를 Softmax layer라고 합니다.
마지막 층의 신경망에서는 선형 부분을 산출할 것입니다.(z[L] = W[L]a[L-1]+b[L]) z[L]를 구했으면 Softmax 함수를 적용해야합니다.
첫번째로 임시적인 분산을 산출할 것입니다. 이 값을 t라고 부르며, 이 값은 e의 z[l]승입니다. 여기서 z[L]은 우리 예제에서 z[L]이 4x1이 될 것입니다. t는 e의 z[l]승인데 이것은 요소별 지수입니다. T또한 4x1차원 벡터입니다. 그리고 결과값 a[L] 기본적으로 벡터 t이고 정규화되어 합이 1이될 것입니다. 그러면 a[L]은 e의 z[L]승 나누기 j가 1에서 4까지의 합입니다. 
이전 이진분류에서는 activation 함수가 single row의 값을 입력으로 받았지만, Softmax Regression에서는 입력도 벡터의 값이며, 결과값도 벡터로 나타납니다.

이제는 은닉층이 없는 신경망의 예시를 알아보겠습니다. 입력으로 x1,x2를 가질 때, z[1] = W[1]x+b[L] / a[1] = y_hat = g(z[1])와 같이 계산될 것입니다.  C = 3일 때, 아래와 같은 Decision Boundary를 가질 수 있습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/191f4750-ae9e-435f-b08b-dbc2b593fbd8)

### Training a softmax classifier
Softmax layer를 사용하는 신경망을 어떻게 훈련할 수 있는지에 대해 배우겠습니다.
hardmax의 경우에는 [1 0 0 0] 처럼 값을 가지는데, 가장 큰 확률을 가지는 요소가 1이 되고 나머지는 0이 되는 반면에, Softmax는 각 클래스의 확률을 값으로 가지며, Softmax Regression은 C개의 클래스를 가진 Logistic Regression을 일반화한 것이라고 할 수 있습니다. 만약 C=2라면, Softmax Regression은 Logistic Regression과 동일한 것입니다.
이제 실제로 softmax layer로 어떻게 학습시키는지 알아보겠습니다. 신경망을 훈련시키는 손실함수를 정의해보겠습니다. ground truth 레이블이 [0 1 0 0] 인 것을 보겠습니다. 신경망이 y_hat을 출력하고 있으므로 결과값은 [0.3 0.2 0.1 0.4]입니다. 이러한 결과라면 신경망이 잘 예측하지 못한것입니다. 손실 값을 줄이려고 한다면 y_hat2를 최대한 크게 만들어야 합니다. 
마지막으로 softmax에서 어떻게 경사하강법을 적용하는지 알아보겠습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/12ce82c3-8cfc-4a2f-9b66-fb08243052f7)

이러한 신경망이 있을 때, output layer의 결과값은 순전파를 통해서 y_hat을 구하고 구한 값을 사용해 손실함수를 계산합니다. 여기서 역전파와 경사하강법을 구해야하는데 역전파를 위해 초기화해야하는 값은 output layer에서 z의 미분항이며 dz[L] = y_hat - y로 정의할 수 있습니다.
