Discover and experiment with a variety of different initialization methods, apply L2 regularization and dropout to avoid model overfitting, then apply gradient checking to identify errors in a fraud detection model.

## 학습목표
- Give examples of how different types of initializations can lead to different results
- Examine the importance of initialization in complex neural networks
- Explain the difference between train/dev/test sets
- Diagnose the bias and variance issues in your model
- Assess the right time and place for using regularization methods such as dropout or L2 regularization
- Explain Vanishing and Exploding gradients and how to deal with them
- Use gradient checking to verify the accuracy of your backpropagation implementation
- Apply zeros initialization, random initialization, and He initialization
- Apply regularization to a deep learning model

### Train / Dev / Test sets
Training Set, Dev Set, Test Set을 잘 설정하면 좋은 성능을 갖는 NN을 빨리 찾는데 도움이 될 수 있습니다.
NN에서는 다음과 같이 결정해야 할 것들이 있습니다.
- layers
- hidden units
- Learning Rates
- Activation Functions
처음 새롭게 NN을 구현할 때에는 적절한 하이퍼 파라미터 값들을 바로 선택하기는 거의 불가능합니다. 그래서, 반복적인 과정을 통해 실험하면서 하이퍼 파라미터 값들을 바꿔가며 더 좋은 성능을 갖는 NN을 찾아야 합니다.

데이터를 Train/Dev/Test Set를 적절하게 설정하는 것이 매우 중요합니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/bdf3d095-0855-4fda-8a31-6f699d2b9324)

이러한 데이터가 있다고 할 때, 일부분을 Training/Dev/Test Set으로 분리할 수 있습니다.
예전에는 train과 test 셋을 7:3으로 분할하는 것이 일반적이었습니다. 명시적인 개발세트가 없거나 기계학습에서는 Training/Dev/Test Set를 6:2:2로 분할하였습니다. 하지만 현대의 빅데이터시대에는 Dev Set과 Test Set은 전체 Data에서 작은 비율을 차지합니다. Dev Set이나 Test Set은 알고리즘을 평가하기에 충분하기만 하면 되기 때문에, 백만개의 Data가 있을 때, Dev Set과 Test Set은 10000개 정도면 충분합니다.

Test Set가 없어도 무방합니다. Test Set의 목표는 편견없는 추정치를 제공하지만(선택한 네트워크의 최종 네트워크 성능), 하지만 그 편견없는 추정이 필요하지 않는다면 테스트 세트가 없어도 괜찮습니다.

### Bias / Variance
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/f9974cc3-c0c6-4f5f-a549-1ab49db997c4)
첫 번째 high bias 그래프는 데이터가 잘 fitting 되지 않아서 underfitting의 문제를 가지고 있습니다. 반면에, 세 번째 high variance 그래프를 살펴보면 아주 복잡한 분류기(ex, DNN)에 fitting 시켰기 때문에, 데이터를 완벽히 fitting시켜서 overfitting 문제를 가지고 있습니다.

만약 Traini set error가 1%이고, Dev set error가 11%인 경우를 살펴보겠습니다. 이 경우에는 Training set에서는 아주 잘 동작하지만, Dev set에서는 잘 동작하지 못하는 것으로 볼 수 있습니다. 그렇다면 Training set에 overfitting했다고 할 수 있으며, 즉, 일반화하지 못했다고 볼 수 있고, High Variance문제를 갖고 있다고 할 수 있습니다.
두 번째로 Train set error가 15%, Dev set error가 16%라고 해보겠습니다. 이 경우에 Human error가 0%의 오류라고 한다면, 해당 알고리즘은 training set과 dev set 모두 잘 동작하지 못하고, 학습 알고리즘이 data에 잘 fitting 되지 않은 경우로 underfitting 문제라고 볼 수 있습니다. 그래서 이 알고리즘은 High Bias 문제를 갖고 있다고 할 수 있습니다.
세 번째로 Train set error가 15%, Dev set error가 30%라고 해보겠습니다. 이 경우에는 training set에서 잘 수행되지 않고 bias가 높기 때문에 높은 bias로 진단합니다.  그리고, Dev set error과 train set error의 차이가 크기 때문에 trainig set에 overfitting해서 High variance 문제도 갖고 있는 최악의 경우라고 볼 수 있습니다.
네 번째로 Train set error가 0.5%, Dev set error가 1%라고 해보겠습니다. 이럴 경우 낮은 bias와 variance를 가지게 됩니다. 하지만, 이와 같은 분석은 Human error가 0%에 근접하거나 또는 Optimal Error(Bayes Error)가 거의 0%에 근접한다는 가정이 있어야지 올바르다고 할 수 있습니다.

만약 Optimal error or Bayes error가 15%라면, 두 번째 분류기의 경우는 사실 그리 나쁜 것은 아니고, High bias로 판단하지도 않을 것입니다. 즉, 흐릿한 이미지가 있어서 사람도 잘 분류하지 못하고 시스템도 분류하지 못하는 사진이 있는 경우에는 조금 다를 수 있고, bias/variance 분석 방법이 다를 것입니다. train set error를 살펴보면서 데이터가 얼마나 잘 fitting 하는지 확인하고, dev set에서 얼마나 variance problem이 존재하는지 확인해야합니다.

![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/c03d2632-79ae-4020-b592-fa465c025033)

직선분류기는 데이터를 underfitting 하기 때문에 high bias를 가지고 있습니다. 만약 분류기가 오작동하여 실제로 데이터의 일부도 과적합 됩니다. 
곡선함수나 이차함수로 분류한다고 해도 너무 많은 유연성이 있기 때문high variance 문제가 발생할 수 있습니다.

### Basic Recipe for Machine Learning
High bias가 있는지 확인하기 위해서는 training set나 data 성능을 살펴봐야 합니다. High bias가 있다면 training set에 잘 맞지 않거나 더 오래 훈련할 수도 있습니다. 이것을 해결하기 위해서는 더 큰 network나 다른 구조의 NN을 선택하거나, 더 길게 학습을 진행하거나, 더 최적화된 알고리즘을 선택하는 방법이 있습니다.
High variance가 있다면 그 문제를 해결하는 가장 좋은 방법은 더 많은 데이터를 얻는 것입니다. 또한 정규화를 시도해볼 수도 있습니다. 적합한 신경망 구조를 사용하는 것도 도움이 된며, 이 경우에는 high bias 문제도 해결할 수 있습니다.
다양한 방식을 시도해보고 낮은 bias와 variance를 가진 것을 찾을 때까지 계속 노력해야합니다.

첫번째로, High bias, High variance 있는지 여부에 따라 시도해야하는 항목은 상당히 다를 수 있습니다. 일반적으로 training set를 사용하여 bias와 variance 문제가 있는지 확인한 후 적절한 하위 집합을 선택합니다. 만약 High bias가 있다면 더 많은 training data를 얻는 것은 실제로 도움이 되지 않습니다. 그러므로 가장 유용한 항목을 선택하는 것이 중요합니다.
두번째로, 머신러닝 초기에는 bias와 variance trade off에 대한 많은 논의가 있었습니다. 시도할 수 있는 여러가지 방법 중에 bias를 늘리면서 동시에 variance을 줄이거나 bias를 줄이거나 variance를 늘릴 수 있습니다. 하지만 딥러닝 이전에 시대에는 도구가 많지 않았기 때문에 한 가지 요소의 희생이 없이는 bais나 variance를 줄일 수 있는 방법이 딱히 없었습니다. 하지만 현재 딥러닝은 빅데이터시대에 맞게 더 큰 네트워크를 계속 훈련할 수 있고 더 많은 데이터를 계속 얻을 수 있는 한 variance를 손상시키지 않으면서 bias를 줄입니다. 이러한 이유때문에 딥러닝이 굉장이 유용하기도 합니다.

### Regularization
정규화는 variance를 줄이기 위해 더 많은 훈련 데이터를 얻기 위한 것이기도 합니다. 정규화를 추가하면 과적합을 방지하는데 도움이 됩니다.
로지스틱회귀의 경우 비용함수 J를 최소화하려고 한다는 점을 기억해야 합니다. 여기에 정규화를 추가하면 람다를 추가해야합니다.(w의 norm에 아래첨자 2는 이 norm이 L2 regularization이라는 의미입니다.)
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/2861cfe4-3b0e-4eac-8446-899bb88f6892)

매개변수 w만 정규화하는 이유는 무엇일까요? b에도 정규화를 진행할 수 있지만 생략하겠습니다. 매개변수를 보면 w는 일반적으로 매우 높은 차원의 매개변수 벡터이며 high variance 문제가 있습니다. 반면 b는 그저 단수일 뿐입니다. 그래서 거의 모든 매개변수들이 b보다는 w에 있습니다.
L2정규화는 가장 일반적인 유형의 정규화입니다. L1정규화를 사용한다면 w 벡터 안에 0값이 많이 존재하게 되어서 w는 결국 희박해집니다. 파라미터 set이 0이고, 모델을 저장하는데 더 적은 메모리를 사용해서, 모델을 압축하는데 유용하기도 합니다. 하지만, 그렇게 많이 사용되지는 않고, 대체로 L2 regularization이 많이 사용됩니다.
람다는 정규화 파라미터입니다. 일반적으로 dev set를 사용하여 이것을 설정하거나 hold out 교차검증을 진행합니다. 그러므로 조정해야하는 하이퍼 파라미터입니다.
파이썬 내부에 lambda라는 함수가 있기 때문에 정규화 매개변수를 나타내기 위해 lambd를 사용합니다.

Neural network의 경우 정규화함수를 추가한 비함수는 다음과 같습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/be0d00cc-b61c-4549-bf81-05bc8ea2c203)

여기서 w는 (n[l], n[l-1]) 행렬입니다. 여기서 사용된 정규화 방식은 Frobenius입니다.

이것으로 경사하강법을 어떻게 구현할까요?
역전파를 통해 dw를 구했고 W[l] := w[l] - αdW[l]의 방식으로 파라미터 W를 업데이트했습니다. 여기서 정규화항을 추가했기때문에 아래와 같은 방식으로 계산됩니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/e0732c59-d810-4ba3-9e40-13a576896ec5)

그리고 파라미터는 아래와 같이 업데이트 됩니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/d3b6f2ec-06d4-4821-af93-b66c12848195)

이러한 이유로 L2정규화를 가중감소라고도 합니다.

### Why Regularization Reduces Overfitting?
정규화는 high variance 문제 즉 overfitting문제를 해결하는데 어떻게 도움이 될까?
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/3567d5da-daca-4fc8-8902-82d2ee1f6f45)

신경망이 제일 오른쪽 그림처럼 overfitting한다고 가정합니다. 그리고 비용함수 J는 다음과 같이 정의됩니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/6a77e27b-df1b-4b09-ba56-e10ce8e94f65)

L2 정규화는 가중치 행렬이 너무 크면 줄여주는 역할을 하는데 이것이 과적합을 방지하는 이유가 됩니다.
람다를 크게 만들면 W를 0에 가깝게 설정하도록 유도할 수 있습니다. 이렇게 된다면 hidden unit의 영향력을 없앨수 있습니다. 그렇게 되면 단순한 신경망은 훨씬 더 작은 신경망이 됩니다.
결과적으로 오른쪽 그림에서 왼쪽의 경우처럼 이동하게 됩니다. 이상적으로 람다의 적당한 중간값을 통해서 딱맞는 값을 도출하면 됩니다.

한가지 예제를 더 살펴보면 아래와 같은 함수가 있다고 할때, 
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/4da75433-1e7d-4fe3-9c92-e64b40dd983e)

이 경우 Z의 값이 작으면 tanh의 선형부분만 사용할 것입니다.
정규화에서 람다값이 크다면 W[l]값이 작아지고 결국 Z[l]값도 작아질 것입니다. 결과적으로 모든 층이 선형인 것과 비슷해집니다.
Activation이 linear하기 때문에 비선형 decision이 불가능해서, overfitting할 수 없게 됩니다.

### Dropout Regularization
L2 정규화와 더불어 Drop out이 있습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/61e548f5-838f-418b-a443-2c9a587e4e92)

이러한 신경망을 training하는데 과적합이 존재할 때 drop out을 통해 신경망의 노드를 제거하는 확률를 설정할 것입니다. 각 층, 노드별로 노드를 유지할 것인지 제거할 것인지에 대한 확률을 0.5로 설정하면 절반의 노드가 제거될 것입니다. 그 후 순전파와 역전파를 진행하면 됩니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/d06ac8cb-5f9e-4de9-9928-d6b109edfb21)

방법이 무작위로 이루어지지만 정상적으로 잘 동작합니다. 훨씬 더 작은 네트워크를 트레이닝 시키기 때문에 정규화 효과를 얻을 수 있습니다.

dropout을 도입하는 방법은 몇가지가 있는데 inverted dropout이 가장 효과적으로 사용됩니다.
예시로 신경망에서 3번째 layer에서 dropout의 적용을 살펴보겠습니다.
1. Dropout vector d를 생성합니다.
d3는 np.random.rand(a3.shape[0], a3.shape[1])를 의미합니다.
2. 그리고 우리는 d3가 어떤 숫자(keep_prob)보다 작은지 확인합니다. 여기서 keep_prob = 0.8이라고 하면, 그렇다면 노드가 제거될 확률이 0.2가 되는 것입니다.
d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob
이렇게 한다면, d3는 0.8의 확률로 1이 되고, 0.2의 확률로 0인 vector가 됩니다.
3. a3를 d3의 요소곱을 합니다. 
a3 = np.multiply(a3, d3)
4. 마지막으로 keep_prob로 나누어줍니다. 
a3 /= keep_prob
만약 50개의 units이 있는 layer라면, 10개의 units은 제거되어서 0이 될 것입니다.
z[4] = w[4]a[3] + b[4]이고, a[3]가 20% 감소한다는 것을 알 수 있습니다.
z[4]의 기대값을 감소시키지 않기 위해 a3에 0.8을 나누는 것입니다. 그러면 a3의 기대값도 변하지 않고 z4의 기대값도 변하지 않을 것입니다.
invert dropout의 기대효과는 keep.prob을 어떻게 설정하더라도 1로 설정되면 dropout은 없습니다.

주의해야 할 점은 test set에서는 dropout을 사용하지 않는 것입니다. 테스트에서 예측할 때 dropout을 사용하면, 예측값이 랜덤하게 되고, 결국에 예측값에 noise만 추가될 뿐입니다. 그리고 keep_prob를 나누어주는 연산을 했기 때문에, 테스트에서 dropout을 사용하지 않더라도 테스트의 기대값은 변하지 않습니다.

### Understanding Dropout
Dropout은 무작위로 unit을 제거하는 역할을 하는데, Regularization이 어떻게 잘 동작할 수 있는 것일까요?
첫번째로 더 작은 신경망을 사용하는 것 정규화 효과가 있는 것 처럼 보입니다. 두번째 단일 유닛의 관점에서 보면 4개의 입력이 있을 때, 의미있는 결과값을 생성해야합니다. 이제 dropout으로 입력이 무작위로 제거될 수 있습니다. 이 때 의미있는 결과값이 되기 위해서는 어떤 한가지 feature에 의존하면 안됩니다. 그것이 언제든지 임의로 제거될 수 있기 때문입니다.그래서 한가지 입력에 큰 가중치를 주는 것을 주저하고 분산시키게 됩니다. 가중치를 분산시켜주면 norm을 축소시키는 효과가 있습니다. L2 Regularization와 유사하게 가중치를 줄이고 과적합 문제를 해결해줍니다.

![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/0248b7fd-30f9-4aa6-aafc-9f5525f48558)

이러한 NN이 있다고 가정합니다. 여기는 3개의 입력 특성이 있는 네트워크입니다. 선택해야하는 것 중 하나는 keep prob이며 레이어마다 유닛을 유지할 확률을 나타냅니다. 따라서 레이어별로 keep prob을 다르게 하는 것도 가능합니다.
첫번째 레이어의 경우 w[1]은 7x3입니다. 두번째 레이어의 경우 w[2]는 7x7입니다. w[3]은 3x7이 됩니다. 그러면 두번째 레이어가 가중치가 가장 큰 행렬입니다. 그래서 그 행렬의 과적합을 줄이기 위해 두번째 레이어의 keep prob이 비교적 낮을 수 있습니다.
Dropout은 Computer Vision에서 도입되었습니다. 컴퓨터비전에서는 입력값의 크기가 너무 커서, 모든 픽셀들을 입력하는데 항상 데이터가 부족합니다. 그렇기 때문에 컴퓨터 비전에서 자주 dropout이 활용됩니다.
Dropout은 정규화 기법이며 overfitting 방지에 도움이 되는데 그 외에는 사용하지않는걸 권유합니다.
Dropout을 사용하게 되면 Cross-validation에서 설정해야할 하이퍼파라미터가 증가한다는 것이 단점입니다.
또한, Cost Function J가 명확하게 정의되지 않는다. 매 반복마다 노드를 무작위로 제거하기 때문에, Gradient Descent 성능을 더블체크하기가 어렵습니다.

### Other Regularization Methods
앞서 배웠던 L2 정규화와 dropout뿐만 아니라 다양한 정규화방법이 있습니다. 
만약 데이터가 비싸고, 더이상 수집할 수 없는 경우에 Data augmentation 방법을 사용할 수 있습니다.이미지를 가로로 뒤집거나, 회전하고 줌인하거나, 또는 찌그러뜨리거나 확대를 해서 Data set을 확장할 수 있습다. 다만, 새로운 것 데이터를 추가하는 것보다는 좋지는 않지만 시간을 절약해주는 장점이 있습니다. 이 방법은 큰 비용없이 overfitting을 줄이는 방법입니다.

**early stopping**
Early Stopping은 Gradient Descent를 실행하면서 training set error나 J function과 dev set error 그래프를 그려서 W가 middle-size일 때, 중지시키는 방법입니다.
학습 초기에 파라미터 W를 0에 가깝게 작은 값으로 초기화를 하고, 학습을 하면서 W는 증가하게 되는데 W의 비율이 중간정도되는 지점에서 중지하게 됩니다.

머신러닝의 단계는 아래와 같이 두 가지로 나누어집니다.
1. Cost Function J를 최적화(Optimize cost function J) - Gradient Descent
2. Not Overfit - Regularization

Early Stopping을 사용하면, 위 두 단계가 함께 적용되고, 별개의 문제를 각각 따로 해결할 수가 없게 됩니다.
즉, overfitting을 해결하기 위해서 Gradient Descent를 중간에 멈추기 때문에, 최적화를 완벽하게 하지 못하고 J를 더이상 줄이지 못하게 됩니다. 이렇게 된다면 이후에 시도할 방법들을 더 복잡하게 만들게 됩니다.

그래서 Early Stopping 대신, L2 Regularization을 사용하는 경우에는 더 길게 학습하면되고, 여러 값의 Lambda를 테스트해야하지만 하이퍼파라미터를 분해하는데 도움이 됩니다.
다만, Early Stopping은 Gradient Descent 한번으로 W값을 설정할 수 있습니다.
그래서 단점이 있지만 종종 사용되는 방법이지만, 주로 L2 Regularization을 사용하는 것을 선호합니다.

### Normalizing Inputs
학습속도를 높일 수 있는 방법 중 하나는 표준화를 하는 것입니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/1706623e-f73a-474b-ab2c-cb771f0de80f)

두 개의 input이 있는 경우를 살펴보면 표준화를 하는 방법은 두단계로 이루어집니다. 
첫번째로 평균을 빼거나 0으로 만드는 단계입니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/1e795b8e-1d43-4d2a-a55c-255911dcda70)

이와 같이 평균을 구해서 x:=x−μ를 진행하면 됩니다.
두번째로 분산을 정규화하는 단계입니다. 특징 x1은 특징 x2보다 분산이 훨씬 큽니다. 정규화를 진행하면 분산이 모두 1이 되는 결과가 됩니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/8bb4f3df-8222-4928-9f42-2fd99fc62e0a)

만약 이 방법을 training set에 적용할 것이라면 동일한 μ,σ값을 test set에 적용해야 합니다.

입력 특징을 정규화하는 이유는 다음과 같습니다. 정규화되지 않은 입력 특징을 사용하는 경우 비용 함수는 아래와 같은 형태를 띄고 있을 수 있습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/b18fb9d3-3eaa-48f9-ba1a-dbd85926267c)

만약 x1의 범위가 1~1000이고, x2의 범위가 0부터 1이라 가정하면 w1과 w2의 범위가 매우 다른 값을 띄게 될 것입니다.
이 값을 정규화시킨다면 아래 그림과 같이 평균적으로 더 대칭으로 보일 것입니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/2babe3c4-a2cd-4b63-a60f-7839072b9c14)

만약 normalization되지 않은 feature를 사용한다면, 매우 작은 learning rate를 사용해야 됩니다. 왜냐하면 Gradient Descent 를 수행하면 더 많은 단계를 거쳐서 최소값에 도달하기까지 계속 왔다갔다 할 수 있기 때문입니다.
물론 실제로는 파라미터 W가 고차원의 matrix이기 때문에 그래프로 나타내기는 힘들어서 정확하게 전달되지 않는 부분도 있지만, 주로 Cost Function은 더 구형이 띄고 feature를 유사한 scale로 맞추게 되면 더 쉽게 최적화할 수 있습니다.
만약 x1이 0~1의 범위를 갖고, x2가 -1~1의 범위를 갖고, x3가 1~2의 범위를 갖는다면, 이 feautre들은 서로 비슷한 범위에 있기 때문에 실제로 잘 동작합니다. 정규화를 진행하는 건 범위가 크게 다를 때가 문제인데, 범위가 비슷할 경우 평균을 0으로 만들고, 분산을 1로 만든다면 학습이 더 빨라질 수 있습니다.

### Vanishing / Exploding Gradients
DNN을 학습할 때 가장 큰 문제점 중 하나는 데이터 소실과 기울기 폭발의 문제가 발생하는 것입니다. DNN을 훈련하는 경우 분산이나 기울기가 때때로 매우 크거나 굉장히 작은 값을 갖게 될 수 있는데 이런 경우 훈련이 매우 까다로울 수 있습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/ae97e9e4-c26a-4999-8469-56487b044dc1)

이렇게 깊은 신경망이 있다고 할 때, 
