Analyze the key computations underlying deep learning, then use them to build and train deep neural networks for computer vision tasks.

## 학습 목표
- Describe the successive block structure of a deep neural network
- Build a deep L-layer neural network
- Analyze matrix and vector dimensions to check neural network implementations
- Use a cache to pass information from forward to back propagation
- Explain the role of hyperparameters in deep learning
- Build a 2-layer neural network

#### Deep L-layer Neural Network
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/7f4b00eb-f26a-4579-8547-2dc733040459)

로지스틱 회귀는 매우 얕은 모델인 반면 5개의 hidden layers로 구성되어 있는 모델은 훨씬 더 깊은 모델이라고 할 수 있습니다. 얕기와 깊이는 정도의 문제라고 말할 수 있습니다.

![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/0aaad74c-18e9-4b74-b05a-2edad5d7bf38)

이 레이어는 4 layer NN이고, 숨겨진 층의 단위 수는 5,5,3이고 하나의 상부 유닛이 있습니다. 사용할 표기법은 대문자 L을 사용하여 네트워크의 레이어 수를 나타냅니다. 이 경우에는 L=4가 됩니다. 그리고 노드 수 또는 레이어 소문자 l의 단위 수를 나타내기 위해 n[l]이라고 표기할 것입니다. 각 레이어 L에 대해 활성화를 나타내기 위해 a[l]이라고 표기할 것입니다. 그래서 나중에 전파를 위해 a[l]을 활성화 g(z[l])로 계산하고 활성화는 레이어 l에 의해 인덱싱 될 것입니다. 그런 다음 W[l]을 사용하여 레이어 l에서 z[l]값을 계산하기 위한 가중치를 나타내고 마찬가지로 b[l]은 z[l]을 계산하는데 사용됩니다. 입력기능은 x라고 불리지만 x는 레이어 0의 활성화이기도 하므로 a[0] = x이고 최종 레이어의 활성화인 a[L] = y_hat입니다. 따라서 a[L]은 신경망의 예측 y_hat에 대한 예측된 출력과 같습니다.
표기법을 최종적으로 정리하면 다음과 같습니다.

![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/17ba813a-1db0-40ae-a34b-e330f5af7def)

#### Forward Propagation in a Deep Network
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/db84e5e2-6847-4e56-ab09-17e09eb012e2)

이렇게 만들어져있는 NN모델서 layer 1의 활성화는 다음과 같이 계산됩니다. 
z[1] = w[1]x + b[1] (w1과 b1은 layer 1의 활성화에 영향을 미치는 매개변수입니다.) 그 레이어에 대한 활성화를 a = g(z[1])과 같도록 계산합니다. 활성화 함수 g는 현재 레이어 및 layer 1의 활성화 함수로서 설정된 인덱스에 따라 달라집니다.
layer 2의 경우 z[2] = w[2]a[1] + b[2]를 계산합니다. 그리고 활성화는 y행렬에 layer 1의 출력을 곱한 것입니다.(a[2] = g[2](z[2])) 이렇게 layer 4까지 계산하면 됩니다.
여기서 확인해야할 것은 x는 a0과도 같다는 것입니다. 입력값 x도 layer0의 활성화이기 때문입니다 그래서 z[1]을 계산할 때 w[1]a[0]을 넣으면 모든 방정식은 동일해지게 됩니다.
최종적으로 정리하면 z[l] = w[l]a[l-1] + b[l] / a[l] = g[l](z[l])이 되고, 일반적인 순방향 전파 방정식이 됩니다.
그리고 벡터화를 진행하게 되면 Z[1] = W[1]A[0] + b[1], A[1] = g[1](Z[1]) / Z[2] = W[2]A[1] + b[2], A[2] = g[2](Z[2]) 가 됩니다. 표기법을 정리하면 Z[l] = W[l]A[l-1] + b[l] / A[l] = g[l](Z[l])가 됩니다.
1-layer에서 반복문을 피해서 벡터화하여 연산을 진행했지만, 각 레이어는 반복문을 쓰는 것 말고는 방법이 없습니다. 따라서, 순전파를 수행할 때, 각 Layer에 대해서 반복문을 수행해야 합니다.
