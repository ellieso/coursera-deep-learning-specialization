Streamline and optimize your ML production workflow by implementing strategic guidelines for goal-setting and applying human-level performance to help define key priorities.

## 학습목표
- Explain why Machine Learning strategy is important
- Apply satisficing and optimizing metrics to set up your goal for ML projects
- Choose a correct train/dev/test split of your dataset
- Define human-level performance
- Use human-level performance to define key priorities in ML projects
- Take the correct ML Strategic decision based on observations of performances and dataset

## Introduction to ML Strategy
### Why ML Strategy
머신러닝 시스템을 더욱 빠르고 효율적으로 작동시키는 방법을 배웠으면 합니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/1465b6d9-ae2c-4770-922f-650cf572ce8d)

고양이를 판별하는 분류기가 있다고 가정할 때, 시스템 정확도가 90%가 되었다. 시스템을 어떻게 개선할지에 대한 다양한 방법이 있습니다.
train data를 더 많이 수집하거나 train set가 아직 충분히 다양하지 않기 때문에 더 다양한 포즈의 고양이 사진이 더 다양한 부정 예시 세트를 수집해야한다고 얘기할 수도 있습니다. 경사하강법을 이용해 알고리즘을 더 오래 훈련시키는 방안이나 아담 최적화 알고리즘 등 다른 최적화 알고리즘을 시도해 볼 수도 있습니다. 아니면 더 큰 네트워크나 더 작은 네트워크를 시도하거나 드롭아웃 또는 L2 정규화를 시도해볼 수도 있습니다. 또는 활성화 함수나 숨은 유닛의 개수 등 네트워크의 구조를 바꿀 수도 있습니다.
이렇게 딥러닝 시스템을 개선하려고 할 때 시도할 수 있는 것들이 굉장히 많은데 선택을 잘못하게 되면 실제로 6개월 정도를 프로젝트 방향만 바꾸는데 소비할 수 있습니다.
이번 강의에서는 가장 시도할만한 방법을 선택할 수 있도록 방향성을 제시하는 머신러닝 분석방법에 대해서 다룰 것이고, 이 과정을 통해서 딥러닝 시스템을 조금 더 효율적으로 접근해서 구현할 수 있을 것입니다.

### Orthogonalization
머신러닝 시스템을 구축할 때 어려운 점 중 하나는 시도해 볼 수 있는 것도, 바꿀 수 있는 것도 아주 많다는 것입니다. 예를 들어 튜닝할 수 있는 하이퍼파라미터가 아주 많습니다. 머신러닝 업무를 가장 효과적으로 수행하시는 분들은 무엇을 튜닝할 것인지에 대한 뚜렷한 안목을 가지고 있습니다. 이러한 절차를 직교화라고 부릅니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/e2ae864d-c7ce-4b42-a16f-07b49bbbcfdf)

이렇게 오래된 TV가 있으면 조정을 통해 사진을 바꿀 수 있는 여러 개의 손잡이가 있습니다. 1개의 손잡이로는 화면의 높이를 조절하고, 다른 손잡이 하나는 너비를 조절하고, 또 다른 손잡이는 사다리꼴 모양의 정도를 조절하고, 다른 손잡이는 화면 좌우 위치를 조절하고, 다른 하나는 화면의 회전 정도를 조절할 수 있을 것입니다. 반대로 특정 손잡이가 높이와 너비, 사다리꼴 정도를 일정 비율로 모두 튜닝한다고 해보겠습니다. 이러한 손잡이가 있다면 높이, 너비, 사다리꼴의 정도 등의 여러 변수들이 한번에 적용되어 한번에 바뀌게 되어서, 원하는 화면(가운데 위치한 사각형)으로 튜닝하기가 거의 불가능할 것입니다. 이런맥락에서 직교화는 TV디자이너들이 각가의 손잡이가 1가지 기능만 갖도록 설계한 것을 의미합니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/3c45e686-8ee3-464b-ae44-3123cf042e54)

또 다른 예시로는 자동차가 있습니다. 운전을 배울 때를 생각해보면 자동차에는 대표 컨트롤이 세개가 있습니다.(Steering, Acceleration, Braking) 이렇게 3가지 컨트롤을 통해 또는 운전대 컨트롤 한개 속도 컨트롤 두 개를 통해 상대적으로 해석이 쉬워져 서로 다른 컨트롤을 이용한 다른 액션을 통해 차가 어떻게 움직일지 알 수 있습니다. 조이스틱식으로 차를 설계한다고 할 때 특정 명령에 0.3x스티어링 각도, -0.8x속도 이런 식으로 조절이 된다고 생각해보겠습니다. 그리고 다른 버튼은 2x스티어링 각도, 0.9x속도를 조절한다고 하면, 스티어링과 속도를 별개로 조절하는 것보다 차를 조정하는 것이 훨씬 더 어려울 것입니다. 따라서, 일차원적으로 3개의 기능이 각각 따로 스티어링, 속도, 브레이크를 담당하는 것이 좋고, 자동차를 원하는 방향으로 조절하기가 더 수월해집니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/df1dc18d-d18a-4100-a866-55a995ea1a3f)

이런것들이 머신러닝과 어떻게 연결되는지 알아보겠습니다. 지도학습이 잘 운영되기 위해선 시스템 손잡이를 조정하여 다음 네가지가 잘 유지되도록 해야합니다. 첫번째로 적어도 Training Set에서는 좋은 성능이 나와야 합니다. Training Set에서의 성능은 수용성 평가를 어느 정도 통과해야 합니다. Training Set에서 좋은 성능이 났다면 이것이 dev set에서도 이어지고 나중에는 test set에서 성능이 잘 나올것입니다. 마지막으로는 비용함수가 적용된 test set에서도 좋은 성과가 나와 시스템이 실제로도 잘 작동해야 합니다. 알고리즘이 Training set에서 비용함수에 잘 맞지 않는다면 더 큰 네트워크를 사용하거나 더 나은 최적화 알고리즘으로 전환할 수도 있습니다. 반대로 알고리즘이 dev set와 잘 안맞는다고 생각되면 정규화를 시도하거나 더 많은 training set를 사용하는 것도 방법이 될 수 있습니다. test set에 잘 맞지 않는다고 하면 더 큰 dev set를 사용해야 할 것입니다. 마지막으로 test set에서는 잘 동작하지만 실제로 잘 동작이 되지 않는다고 하면 dev set를 바꾸거나 비용함수를 변경할 수 있습니다.
신경망을 훈련시킬 때 early stopping을 잘 사용하지 않습니다. 왜냐하면 너무 일찍 학습을 멈추게 된다면 training set에서 잘 동작하지 않을 수 있는데, 동시에 early stopping은 dev set 성능을 향상시키는데 사용되기 때문입니다.

## Set up your goal
### Single number evaluation metric
하이퍼 파라미터를 튜닝하거나 알고리즘 교육에 다른 아이디어를 시도하거나 머신러닝 시스템을 만드는 데 있어 다양한 옵션을 시도하는 등 어떤 경우라도 단일 실수 평가 지표가 있다면 진행속도가 훨씬 빨라질 것입니다. 이 지표는 새롭게 시도한 것들이 더 잘 작동하는지 더 안 좋은 결과를 내는지 알려주기 때문입니다.
Classifier A와 Classifier B가 있다고 해보겠습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/f657bd9c-a8ff-4c98-83d1-fed6f0227830)

여기서 Precision과 Recall 항목이 있는데, 용어의 의미는 다음과 같습니다.
Precision : 정확도, 정밀도를 의미하고, True라고 판별한 것 중에서 실제 True인 것의 비율입니다.
Recall : 검출율, 재현율이라고 하는데, 검출된 결과가 얼마나 정확한지, 즉, 검출된 결과들 중 실제로 실제로 일치하는 것이다. 실제 True인 것들 중에서 True라고 판별한 것의 비율을 의미합니다.

위 표에서는 Precision은 B가 더 뛰어나지만, Recall은 A가 더 뛰어나기 때문에 어떤 Classifier가 더 좋은지 확인하기 어렵습니다. 그러므로 여러가지의 Classifier로 시도해본다면, 테스트해보고 가장 좋은 것을 선택하면 됩니다.
어떠한 방법으로 평가를 해야될지 모르겠을 경우  Precision과 Recall을 결합시킨 새로운 평가지표를 사용하는 것입니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/183903d5-8384-492c-809c-e1f0a2a80fd5)

F1 score의 공식은 2/(1/P+1/R)입니다. Precision와 Recall의 균형을 맞추는데에는 장점이 있습니다. Classifier A가 B보다 더 좋은 F1 Score를 갖는 것을 볼 수 있고, 결과적으로 Classifier A가 더 좋다고 볼 수 있습니다.

![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/21a5f90b-9477-4bc1-ac04-bab3b8542596)

다른 예시를 보면 4개의 대륙에 대한 고양히 분류기를 만든다고 할 때 2가지 Classifier가 있고, 각각 대륙별로 다른 값의 에러를 갖게 된다고 해보겠습니다. 이렇게 4개의 지표를 가지고는 A가 나은지, B가 나은지 결정하기가 쉽지 않습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/b3c11887-b88f-448c-970d-8b2ea1d5d46f)

더 많은 Classifier로 테스트하게 된다면, 이런 결과를 가지고 한가지 알고리즘을 선택하는 것은 너무 어렵게 됩니다. 위와 같이 평균값을 산출함으로써, 알고리즘 C가 가장 낮은 평균의 에러 값을 가지고 있다는 것을 빠르게 확인할 수 있고, 우리는 알고리즘 C를 선택할 수 있을 것입니다.

### Satisficing and Optimizing metric
단일 실수 평가지표로 결합해서 나타내는 것은 쉬운 일이 아닙니다. 이런 경우에는 satisficing과 optimizing이 유용할 수도 있습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/3b982886-f5c1-4b30-b3d5-40b2c6bf0ed8)

이와 같이 고양이 분류기가 있는데 정확도와 실행시간도 고려한다고 하겠습니다. 한가지 할 수있는 것은 정확도와 실행시간을 결합시켜 평가수치를 만드는 것입니다.
다른 방법으로는 정확도를 최대로하는 방법이 있습니다. 실행시간의 최대치는 100ms이내로 설정하고, 이 수치를 만족하는 분류기 중에서 가장 정확도가 높은 것을 선택하는 것입니다. 여기서 정확도는 optimizing metric이라고 하고, 실행시간은 satisficing metric이라고 합니다. 이 방법을 사용하면 B가 가장 좋은 Classifier라고 할 수 있습니다.
일반적으로 N개의 고려해야되는 matrix가 있으면, 1개를 optimizing metric으로 선택하고, 나머지 N-1개를 satisficing으로 선택하는 것이 합리적입니다.

### Train/dev/test distributions
training/dev/test set을 어떻게 설정하는지에 따라 머신러닝 프로젝트 속도가 크게 달라질 수 있습니다. 이 영상에서는 dev/test set을 설정하는 방법에 대해서 알아보도록 하겠습니다.
일반적인 머신러닝의 워크플로우는 다양한 아이디어를 시도해보고 다양한 모델을 training set에서 훈련시켜보고 dev set를 이용해서 여러 아이디어를 평가한 뒤 그 중 하나를 고르는 것입니다. dev set의 성능 개선을 위해 마지막으로 한가지를 선택하고 test set로 평가해보는 것입니다.
US, UK, Other Europe, South America, India, China, Other Asia, Australia에서 고양이 분류기를 운영하고 있다고 하면 어떻게 dev set와 test set를 설정할 수 있을까요? 한가지 방법으로는 언급한 대륙 중에서 4가지를 고를 수 있습니다. 이 4가지 대륙은 무작위로 선별된 대륙일 수도 있습니다. 나머지 4개의 대륙은 test set이 될 것입니다. 하지만, 이렇게 dev set과 test set을 설정하는 것은 dev set과 test set이 모두 다른 분포도를 갖고 있기 때문에 아주 좋지 않은 방법입니다. 그렇기 때문에, dev set과 test set을 설정할 때에는 같은 분포도를 가지는 데이터에서 설정하는 것을 권장합니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/2da44cb3-5985-4b11-95c5-492754c9a0ee)


### Size of the dev and test sets
지난 영상에서 dev set과 test set가 같은 분포에서 와야한다는 것을 배웠습니다. 그렇다면 크기를 어떻게 설정해야하는지에 대해 알아보겠습니다.
머신러닝 초기에는 training set와 test set를 7:3의 비율로 나누는 방법을 사용했었는데 dev set도 사용해야 한다면 6:2:2의 비율로 사용했습니다. 데이터의 크기가 크지 않는 초기에는 합리적인 방법이었습니다. 하지만 최근에는 훨씬 더 큰 데이터의 세트를 다룹니다. 만약 백만개의 데이터가 있다고 할 때, training set를 98%, 나머지를 1%씩 분배하는 것이 이상적이라고 할 수 있을것입니다. 
Test set의 목적은 시스템의 개발이 완료된 후에, 마지막에 시스템이 얼마나 좋은 성능을 가지고 있는지 평가하는데 도움을 줍니다. 최종시스템의 성능을 정확하게 측정할 필요가 없는 이상 많은 test set는 필요없을 수 있으며 있다고 생각하는 수준이면 됩니다. 
