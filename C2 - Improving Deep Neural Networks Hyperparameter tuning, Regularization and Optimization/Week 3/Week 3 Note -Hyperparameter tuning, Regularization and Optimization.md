Explore TensorFlow, a deep learning framework that allows you to build neural networks quickly and easily
, then train a neural network on a TensorFlow dataset.

## 학습목표

- Master the process of hyperparameter tuning
- Describe softmax classification for multiple classes
- Apply batch normalization to make your neural network more robust
- Build a neural network in TensorFlow and train it on a TensorFlow dataset
- Describe the purpose and operation of GradientTape
- Use tf.Variable to modify the state of a variable
- Apply TensorFlow decorators to speed up code
- Explain the difference between a variable and a constant

### Tuning process
신경망을 변경하려면 다양한 하이퍼파라미터를 설정해야하는 것을 배웠습니다. 
 그렇다면 최적의 하이퍼파라미터 세팅을 찾는 방법을 알아보도록 하겠습니다.
Deep Neural Network를 훈련하는 과정에서 힘든 부분 중 하나는 많은 양의 하이퍼 파라미터 개수입니다.
이런 하이퍼 파라미터들은 학습률을 나타내는 α값 모멘텀을 쓰는 경우 모멘텀 값, 아니면 Adam 최적화 알고리즘의 하이퍼파라미터인 β1, β2, ϵ이거나 layer의 수를 선택해야할 수도 있습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/a5de6b9e-5c6a-4222-b221-6155e3c6ffd2)

하이퍼파라미터 중에서 α가 가장 중요하며, 다음으로 노란색 박스의 하이퍼파라미터(β, # of hidden units, mini-batch size)가 중요하고, 다음으로는 보라색 박스인 # of layer, learning rate decay가 중요하다고 할 수 있습니다.

하이퍼파라미터 튜닝을 할 때 어떻게 설정해서 탐색해야 할까요?
머신러닝 알고리즘 초기에는 2개의 하이퍼파라미터가 있는 경우에 점들을 격자판 형식으로 샘플링하는 경우가 많았습니다. 이러한 방법은 하이퍼 파라미터의 개수가
비교적 많지 않은 경우에 잘 동작하는 편입니다.
딥러닝의 경우에 추천하는 방법은 임의로 설정하는 것 입니다. 이러한 방법을 사용하는 이유는 어떤 하이퍼파라미터가 가장 중요할 지 미리 알기가 어렵기 때문입니다.
예를 들어서, 하이퍼파라미터1은 learning rate인 α이고 하이퍼파라미터2는 Adam 알고리즘의 ϵ이라고 해보자.
이런 경우에는 α는 매우 중요하고 ϵ은 거의 중요하지 않을 것입니다. 
따라서, 왼쪽같이 격자판 형식으로 샘플링을 진행하면 25개의 모델을 학습했지만, 5개의 α값으로만 시도할 수 있을 것입니다. 
반대로 샘플링을 무작위로 진행한다면, 각각 25개의 α를 시도하게 됩니다.
