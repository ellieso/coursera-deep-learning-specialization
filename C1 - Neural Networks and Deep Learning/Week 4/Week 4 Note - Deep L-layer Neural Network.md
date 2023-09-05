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

#### Getting your Matrix Dimensions Right
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/d5249af1-d95f-4439-b586-1cef43c80bed)

이 그림은 5 layers NN로 4개의 hidden layer와 1개의 output layer로 구성되어있습니다. 순전파를 구현하는 경우 첫번째로 z[1] = w[1]x + b[1]입니다. 여기서 b는 무시하고 w에 집중하면 첫번째 hidden layer에는 3개의 hidden layer로 구성되어 있습니다. 그러므로 n[1] / n[2]=5 / n[3] = 4 / n[4] = 2 / n[5] = 1이 됩니다.
각 행렬를 구해보면 z[1]는 (3,1)이 될 것이고 이것은 (n[l],m)입니다. x의 행렬은 (2,1)이 될 것이고 이것은 (n[0],m])이 됩니다. w[1]는 (3,2)이 될 것이고 이것은 (n[l], n[l-1])입니다. b[1]은 (3,1)이 될 것이고 이것은 (n[l],1)입니다. 이것은 미분을 진행한다고 해도 동일한 결과를 가져올 것입니다.(dZ[l], dA[l] = (n[l], m))

#### Why Deep Representations?
여러가지 문제에서 Deep Network가 잘 작동한다고 알고 있을텐데요 단순히 신경망의 크기가 크다고 잘 작동하는 것이 아니라 Deep Network는 깊고 숨겨진 레이어가 많아야 합니다.
Deep Network Computing이란 무엇일까요? 얼굴인식이나 얼굴감지를 위한 시스템을 구축하고 있다면 심층 신경망이 할 수 있는 일은 얼굴 사진을 입력한 후 신경망의 첫번째 레이어를 특정 탐지기 또는 가장자리 탐지기로 생각할 수 있습니다. 아래의 예에서는 20개 정도의 숨겨진 유닛을 가진 신경망이 이 이미지에서 무엇을 계산하려고 하는지 플로팅합니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/d2a82725-2bb0-4727-9012-fb85106dfcda)

이 작은 사각형 상자로 시각화된 20개의 숨겨진 유닛이 있습니다. 이제 픽셀을 그룹화하여 가장자리를 형성함으로써 이 그림의 가장자리가 어디에 있는지 생각합니다. 그런 다음 가장자리를 감지하고 가장자리를 그룹화하여 면의 일부를 형성할 수 있습니다. 그런 다음 마지막으로 눈, 코, 귀, 턱 등 얼굴의 다른 부분을 조합하여 다양한 유형의 얼굴을 인식하거나 감지하려고 시도할 수 있습니다. 
즉, 초기의 layer에서는 간단한 모서리와 같은 함수를 감지하는 것이고, 다음 단계의 layer에서 이런 내용들을 취합해서 더욱 복잡한 함수를 배웁니다.
얼굴 인식 분야외에도 음성 인식 시스템도 동일합니다. 음성을 시각화하는 것이 쉽지는 않지만, 이 시스템의 첫 번째 layer에서는 low level wave form을 감지하도록 학습할 수 있습니다.(백색소음인지 등). 그리고 이러한 정보를 취합하여 기본적인 음성 unit(phoneme)을 감지할 수 있습니다.

Deep Network가 잘 작동하는 또다른 이유는 Circuit theory에서 유래된 결과입니다. 이 함수들은 작지만 심층 신경망으로 계산되며 작다는 것은 숨겨진 유닛의 수가 상대적으로 적다는 것을 의미합니다. 그러나 얕은 네트워크로 동일한 함수를 계산하려고 하면 숨겨진 레이어가 충분하지 않다면 계산하기 위해 숨겨진 유닛이 기하급수적으로 더 많이 필요할 수 있습니다.
예를 들어 x1 XOR, x2 XOR, x3 XOR ...  XOR xn까지 계산하려고 할 때, XOR트리에 삽입하면 x1과 x2의 XOR을 계산하고 x3과 x4를 취해서 XOR을 계산합니다. 그런 다음 XOR 트리를 구축할 수 있습니다. 이렇게 해서 최종적으로 Y라고 불리는 회로를 만들 수 있습니다. XOR를 계산하기 위해 네트워크의 깊이는 로그 N의 순서대로 됩니다. 이 네트워크 내의 노드 수, 회선 컴포넌트 수, 게이트 수는 그다지 많지 않습니다. 하지만 여러 개의 숨겨진 레이어가 있는 신경망을 사용할 수 없다면 이 경우에는 로그와 숨겨진 레이어를 주문해야합니다. 하나의 숨겨진 레이어로 함수를 계산해야 한다면 이 모든 것들이 숨겨진 유닛으로 들어가게 해야합니다. 그런 다음 이러한 것들이 Y를 출력합니다. 결과적으로 Y를 계산하기 위해서는 숨겨진 레이어가 기하급수적으로 커야합니다.

#### Building Blocks of Deep Neural Networks
![image](https://github.com/ellieso/coursera-deep-learning-
specialization/assets/83899219/e9b6cab4-9b3f-485e-9f02-dfb5647b0355)

이전 시간에 순전파와 역전파에 대해서 배웠고 이것을 하나로 합쳐서 어떻게 Deep Network를 구성하는지 살펴보겠습니다. 여기 4개의 Layer로 구성된 DNN이 있을 때, 한 Layer에서 순전파와 역전파는 다음과 같이 계산됩니다.
순전파 = input a[l-1], output a[l], Cache z[l]
역전파 = input da[l], output da[l-1], dw[l], db[l]
여기서 cache는 순전파에서 계산한 z[l]이 역전파에서 dz[l]을 구하는데 다시 사용되기 때문에 저장해 놓은 것을 의미합니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/a4e876ea-a7f2-4752-9ff3-100eb8a37e0f)

위를 토대로 전체 DNN에서의 과정을 요약하면 다음과 같습니다.

#### Forward and Backward Propagation
순전파>
input a[l-1]
output a[l], cache(z[l])
