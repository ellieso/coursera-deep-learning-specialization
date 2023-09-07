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


