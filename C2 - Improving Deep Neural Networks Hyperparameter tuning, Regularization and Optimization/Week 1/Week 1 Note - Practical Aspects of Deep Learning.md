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


