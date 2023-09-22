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

### When to Change Dev/Test Sets and Metrics?
어떻게 dev set와 평가지표를 설정하는지 배웠는데 이것은 목표를 이룰 수 있도록 방향성을 제시해주는 것과 같습니다. 목표를 잘못 설정했다는 것을 뒤늦게 깨달을 수도 있는데 이 때는 목표를 변경해야 합니다. 

![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/f71791e2-deb6-43e3-bd47-eed83bc862c7)

고양이 분류기를 만든다고 했을 때 각각의 오차를 가지고 있고 겉보기에는 A가 성능이 더 좋은 것처럼 보입니다. 하지만, 실제로 동작했을 때, 어떤 이유에서인지 알고리즘 A가 부적절한 이미지를 더 많이 허용하고 있는 것을 볼 수 있었습니다. 반대로 알고리즘 B는 부적절한 이미지를 허용하지 않습니다. 그러므로 오차와는 다르게 알고리즘 B가 더 좋은 알고리즘이라고 평가할 수 있습니다. 이런 경우가 발생한다면, 평가 지표를 변경하거나 dev/test set을 바꾸어야 합니다.
기존 에러를 산출하는 식을 보완하기 위해서 가중치를 추가해주어야 합니다. 이미지가 부적절한 경우 10이나 100으로 식을 만드는 것입니다. 

![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/c6934d24-535a-412d-aab6-1d79318512ac)

이 경우에 Orthogonalization을 적용할 수 있는데, 

첫 번째로는 분류기를 평가하기 위한 metric을 정의하는 데에 집중을 하고, 이후에 어떻게 metric을 잘 구현하는 지에 대해서 고민하는 것입니다. 즉, target을 plan하는 것과 tuning 단계를 분리해서 생각하는 것입니다.

## Comparing to human-level performance
### Why Human-level Performance?
시간이 지나면서 머신러닝 프로젝트의 진전은 다음과 같은 경향을 가지고 있었습니다.

![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/805224c8-41fd-45d4-9116-0f074d560d03)


성능이 인간 수준에 근접할수록 빠른 진전이 일어나는 경향이 있습니다. 조금 지나면 알고리즘이 인간수준 성능이 근접하면서 진행속도나 정확도가 정체되기 시작합니다. 그러면서 능숙도가 증가할 수도 있지만 인간 수준 성능을 능가한 뒤에도 지속적으로 더 좋아질 수는 있지만 정확도 상승 속도는 보통 더뎌지게 됩니다. 그러나 알고리즘을 계속 훈련시키면 더 많은 데이터를 더 큰 모델에 사용할 수는 있겠지만 성능은 이론적인 한계점에 다가갈 뿐 절대로 거기에 도달하지는 않습니다. 이를 베이즈 최적 오류라고 합니다.
인간수준성능을 능가하기 전까지 발전속도는 매우 빠르지만 그 이후에 느려집이다. 여기에는 두가지 이유가 있습니다. 첫번째로는 많은 작업에서 인간 수준 성능이 베이즈 최적 오류 지점과 멀리 떨어져 있지 않기 때문입니다. 두번째로 성능이 인간수준에 미치지 못할 경우 여러가지 도구를 이용해 성능을 향상시킬 수 있습니다. 이런 도구들은 인간수준성능을 초과한 시점에서는 사용하기 어렵습니다.

머신러닝 알고리즘이 인간수준 성능에 미치지 못한다면 다음과 같은 방법을 사용할 수 있습니다.

![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/cdfba719-3689-414a-9502-4ed587295a32)

### Avoidable bias
알고리즘이 training set에서 좋은 성능을 내면 좋다고 얘기했었는데 가끔씩은 너무 잘하는 것이 좋지 않을 때가 있습니다.  이때, 인간레벨의 성능이 어느 정도인지 알면, 적당한 수준으로 알고리즘이 training set에서 성능을 발휘할 수 있도록 가능하게 할 수 있습니다.

![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/917ad95b-e9ad-4e52-8efa-7a486ec9000a)

이런 고양이 분류기가 있다고 할 때 training set와 인간의 차이가 매우 크기 때문에 잘 훈련되지 않고 있음을 보여줍니다. 그러므로 편향을 줄이는데 중점을 둘 수 있을 것입니다. 더 큰 신경망 네트워크를 사용하거나 더 긴 시간 학습하는 방법이 있을 것입니다.

![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/aef06b3b-2ef9-440d-86bd-03af1cfc34e3)

이런 고양이 분류기가 있다고 할 때 training set는 인간수준에 조금 못미치는 성능을 가집니다. 이런 경우에는 분산을 줄이는 방향으로 개선할 수 있으며 정규화를 사용할 수 있을 것입니다.

두 예시에서 인간수준성능이 얼마인지에 따라 각각 다른 개선 방법을 사용하고 있습니다.  여기서 Bayes error와 training error의 차이를 avoidable bias라고 부릅니다. 우리는 training error를 bayes error와 가까워지도록 개선하는 것이 목표이며, overfitting 하지 않는 이상 bayes error보다는 낮아질 수 없습니다.

### Understanding human-level performance
인간수준성능이라는 표현은 리서치나 기사에서 쓰이는 단어입니다. 더 자세히 쓰면 인간수준성능이라는 단어는 머신러닝 프로젝트 진전의 촉진제 역할을 할 수 있습니다.

![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/4c9b71fc-845d-41b9-81aa-f24e356e8901)

human-level error는 bayes error를 추정하는 방법이 될 수 있습니다. 이런 영상의학사진을 본다고 할 때, 일반적인 인간은 3% 오류를 내고 영상의학 전문의는 1%의 오류를 내고 경험이 더 많은 의사는 0.7%의 오류를 내고 경험이 더더 많은 의사들로 구성된 사람은 0.5%의 오류를 냅니다. 이런 경우에 human-level error는 어떻게 측정할 수 있을까? 한 가지 접근 방법은 error의 추정치를 bayes error로 접근하는 것입니다.
논문이나 시스템 도입이 목적인 경우에는 사용할 수 있는 인간수준성능이 다르게 정의될 수 있고, 아마 일반적인 의사의 능력 이상만 되어도 될 것입니다. 성공한다면 굉장이 유용하고, 의사 개인의 역량을 뛰어넘은 시스템이라고 한다면, 실제로 도입이 가능할 수 있는 시스템이 되기 때문입니다.

3가지의 예시가 있습니다. 첫번째는 training error가 5%이고, dev error가 6%이고, human level error가 누구냐에 따라서 1%, 0.7%, 0.5%의 값을 가진다고 하겠습니다.  어떤 값으로 지정해도  avoidable bias는 4% ~ 4.5%의 값을 가지고, training error와 dev error의 차이는 1%의 값을 가집니다. 이 예시에서는 인간수준성능은 크게 중요하지 않고 편향을 줄이는 방법에 중점을 두어야 합니다.
두번째는 인간수준성능이 1%, 0.7%, 0.5%의 값을 가지고  training error가 1%, dev error가 5%입니다.  avoidable bias가 0% ~ 0.5%의 값을 가지고, 반면에 training error와 dev error는 4%의 값을 가집니다. 이 경우에는 분산을 줄이는 방법에 중점을 두어야 합니다.
마지막은 training error가 0.7%이고 dev error가 0.8%인 경우입니다. 인간수준성능을 0.5%로 지정하는 것이 중요합니다. 이 경우에는 avoidable error가 0.2%이고, bias를 줄이는 것에 조금 더 중점을 둘 수 있습니다. 만약 0.7%로 한다면 분산을 줄이는 방법에 중점을 둘 수 있습니다. 여기서 머신러닝에서 인간레벨의 성능에 도달해가는 시점에서 더 개선하기가 왜 어려운지를 알려줍니다. 0.7%의 training error에 도달했을 때, bayes error를 정의하는 것에 굉장이 신경쓰지 않는다면, training error가 bayes error에서 얼마나 차이가 있는지 알기는 어려울 것이고, 따라서 우리가 avoidable bias를 얼마나 줄여야 할지도 모르게 됩니다. 추가로, 머신러닝의 성능이 이미 꽤 괜찮은 상황에서 발생했기 때문에 training set를 fitting하기에 매우 까다롭습니다.
인간레벨의 성능과 가까이 있는 경우에는 bias와 variance 효과를 제거하기 더 어렵고, 결과적으로 인간레벨의 성능에 가까워질수록 개선하기가 더욱 어려워지는 것입니다.

### Surpassing Human-level Performance
인간수준성능에 근접하거나 능가한 시점에서는 머신러닝을 개선하기에 매우 어렵습니다.

![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/2da3e397-76ae-42f5-92ee-e1e20fe13e42)

토론하는 문제가 있다고 가정할 때 토론을 통해 위와 같은 오류가 생겼다고 가정해보겠습니다. 이런 경우에 avoidable bias는 0.5%가 bayes error의 추정치가 될 것이고, avoidable bias는 0.1%이고, variance는 0.2%정도로 추정하게 될 것입니다. 이런 결과로 avoidable bias나 variance를 줄일 수 있는 방법을 고려해볼 수 있을 것입니다.

![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/0f456240-4dc9-4f8d-bbaa-9605f8aba937)

이런  training error가 0.3%이고, dev error가 0.4%라고 한다면 avoidable bias 경우라면 bayes error가 0.1 or 0.2 or 0.3%일 수도 있습니다. 정보가 충분하지 않기 때문에, 알고리즘에서 bias를 줄이는데 중점을 두어야할지, variance를 줄이는데 중점을 두어야할지 알기 어려워집니다.

### Improving your model performance
학습 알고리즘을 향상시키는 방법에 대해 알아보겠습니다. 효과적으로 알고리즘이 동작하기 위해서는 두가지 조건을 충족해야합니다.

![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/563fd2ce-94e9-475c-bd7a-cfdbce6f7e24)

첫 번째로는 training set에 잘 fitting되어야 한다는 것이고, 이것은 낮은 avoidable bias를 갖는다는 것과 동일한 의미입니다. 두 번째로는 dev/test set에서도 잘 fitting되어야 한다는 것이고, variance가 나쁘지 않다는 것을 의미합니다. 
즉, 머신러닝의 성능을 향상시키고자 한다면, training error와 bayes error의 차이를 보고, avoidable bias문제를 살펴보고, 그 이후에 dev error와 training error의 차이를 살펴보면 됩니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/0c684193-3137-4e8d-bf32-b03e30159414)

avoidable bias를 줄이는 방법으로는 더 큰 모델을 학습시키거나, 더 긴 시간을 가지고 학습하거나, 더욱 향상된 최적화 알고리즘(Momentum, RMSProp, Adam)을 사용하는 방법이 있습니다. 또 다른 방법으로는 더 좋은 구조를 사용하는 것인데, 더 좋은 hyperparameter를 사용하거나, activation function의 선택 및 layer의 개수, hidden unit의 개수를 바꾸는 방법이 있을 수 있습니다.
variance가 문제라면, 더 많은 data를 수집하거나, regularization을 시도해볼 수 있습니다. 그리고 avoidable bias와 동일하게 NN의 구조를 변경할 수 있고, hyperparameter를 변경해서 개선할 수도 있습니다.

