Develop your deep learning toolbox by adding more advanced optimizations, random minibatching, and learning rate decay scheduling to speed up your models.

## 학습목표
- Apply optimization methods such as (Stochastic) Gradient Descent, Momentum, RMSProp and Adam
- Use random minibatches to accelerate convergence and improve optimization
- Describe the benefits of learning rate decay and apply it to your optimization

### Mini-batch Gradient Descent
신경망 알고리즘을 더 빨리 훈련시킬 수 있는 최적화 알고리즘에 대해 배우겠습니다. 딥러닝을 빅데이터에 훈련시키면 그 속도가 매우 느릴 것입니다. 이럴 경우 빠른 최적화 알고리즘과 양질의 최적화 알고리즘을 갖는 것이 효율성의 속도를 높일 수 있습니다.

일반적인 Gradient Descent는 Batch Gradient Descent라고 부르며, 파라미터를 업데이트할 때 모든 training examples m에 대해서 gradient를 계산한 다음에 파라미터를 업데이트하게 됩니다. 즉, GD를 진행하기 전에 전처리 과정(Gradient 계산)이 필요하고, m이 매우 커질수록 느려지게 됩니다.

전체를 아주 작은 훈련세트로 나눠서 미니 배치들이라고 부르겠습니다. 만약 5,000,000개의 training 샘플이 있고 이 값을 1000개로 미니배치하면 x1~x1000를 x{1} / x1001~x2000을 x{2} ...으로 표현하게 됩니다. 그렇게되면 총 5000개의 미니배치가 생성되게 됩니다. 마찬가지로 y에 대해서도 동일작업을 수행합니다. 미니배치는 X{t}, Y{t}형태의 index를 가지게 됩니다.

미니배치의 진행방식은 다음과 같습니다. trainin set에서 미니배치경사하강볍을 실행하려면 각각 미니배치마다 실행해주면 됩니다.
1. 입력값에 대한 순전파를 도입합니다.(Z[l] = W[l]X{t} +b[l])
2. 활성함수에 적용합니다. A[l] = g[l](Z[l])
3. A[L] = g[L](Z[L])이 될 때까지 시도합니다.
4. 비용함수를 산출합니다.
5. 역전파를 도입합니다. (W[l] = W[l] - αdW[l], b[l] = b[l] - αdb[l])
여기서 한번의 for문 진행을 '1 epoch'라고 하며, 여기서는 한 번의 Gradient Descent에서 총 5000 epoch를 진행합니다. 따라서 전체 iteration을 위한 또 하나의 for loop가 필요합니다.

### Understanding Mini-batch Gradient Descent
경사하강법을 구현하는 방법과 이것이 하는 역할 및 효과적인 이유에 대한 이해를 얻어보겠습니다. 모든 반복에서 배치경사하강법을 사용하여 전체를 훈련시키면 모든 단일반복에서 비용이 감소할 것으로 예상됩니다. 그러면 비용함수 J를 다른 반복함수로 가지면 모든 단일 반복에서 감소해야합니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/d3a574a3-45f7-4a03-8ccc-cbe1a867b8cd)

하지만 미니배치 경사하강법의 경우에 비용함수의 진행률을 그려보면 모든 반복에서 감소하지 않을 수도 있습니다. 아래와 같은 그림으로 감소하는 경향을 보이며 약간의 노이즈도 생길 것입니다. 이렇게 노이즈가 생기는 경우는 어떤 미니배치의 비용이 조금 더 낮을 수도 있고 다른 미니배치의 레이블이 잘못됬을 수도 있습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/d54a73aa-44f9-4dd5-bac0-ff68b5c608f8)

미니배치 경사하강법에서 선택해야 하는 파라미터는 미니배치 사이즈입니다. 만약 mini-batch size가 m인 경우에는 단순 batch GD를 의미합니다. 그리고 mini-batch size가 1인 경우에는 Stochastic Gradient Descent(확률적 경사하강법)이라고 하는데, 각각의 example하나가 mini-batch가 되는 것입니다.
실제로 우리가 사용하는 mini-batch size는 1과 m사이의 값이 될 것입니다. mini-batch size를 m으로 설정한다면, 매 iteration마다 아주 많은 training sets을 처리하게 되고, m이 너무 크다면 긴 시간이 소요됩니다. 만약 m이 작다면 batch GD를 그냥 사용해도 괜찮습니다.
반대로 mini-batch size를 1로 설정한다면, 1개의 example만 처리해도 파라미터를 업데이트할 수 있다는 점에서 괜찮습니다. noise가 많이 발생할 수도 있지만, 더 작은 값의 learning rate를 사용해서 개선시키거나 감소시킬 수 있습니다. 하지만, Stochastic GD는 각각의 sample을 처리한다는 점에서 벡터화의 효율을 잃어버리기 때문에 벡터화를 통한 빠른 계산 속도를 잃는다는 단점이 있습니다.
그러므로 실제로 가장 잘 동작하는 것은 mini-batch size가 너무 크거나 작지 않은 경우입니다. 실제로 가장 빠른 학습 속도를 보여주며, 이러한 경우에 2가지 부분에서 좋은 점이 있습니다.
1. Stochastic GD와 비교해서 벡터화를 통한 속도를 잃지 않습니다.(1000개씩 처리하는 것이 하나씩 처리하는 것보다 빠름)
2. 전체 training set m을 처리하기까지 기다릴 필요없이 파라미터를 업데이트(GD 수행)할 수 있습니다. (한 번의 iteration으로 5000 epoch의 GD step을 수행)

미니배치를 선택하는 방법은 다음과 같습니다.
1. 작은 training set일 경우 배치경사하강법을 해라. 작은 training set의 경우에는 매니배치경사하강법을 쓰는 의미가 없습니다.(n이 2000보다 작을때)
2. 조금 큰 training set의 경우 전형적인 미니배치의 크기는 64~512가 대표적입니다. 컴퓨터 메모리의 방식에 따라 달라지고 2의 지수값을 가질 때 더 빨리 실행됩니다.
3. 미니배치가 cpu,gpu 메모리에 있도록 합니다. 이것은 application이나 training sample의 크기에 따라 변할 수 있지만, CPU/GPU에 들어가지 않는 mini-batch를 처리하는 경우에 성능이 저하되고, 더 악화되는 것을 볼 수 있습니다.

### Exponentially Weighted Averages
지수 가중 이동평균에 대해서 알아보겠습니다. 
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/36741927-9a49-441e-bc7f-7a8cc3b95ba7)

영국의 날씨는 위의 그림처럼 표현할수 있습니다. 이 데이터는 약간 noisy하게 보이지만 온도의 추세, 지역 평균 또는 이동평균을 계산하려면 다음과 같습니다.
일단 V0를 0으로 초기화 합니다. 다음에 일별로 표시되는 항목 0.9배에 해당 날짜 온도 0.1배를 더한 가중치로 평균을 냅니다. 두번째 날에는 가중평균을 적용시킬 것입니다. 결과적으로 보면 V라는 날에 대한 일반적인 공식은 Vt = 0.9Vt-1 + 0.1θt입니다. 이걸 계산하고 빨간색으로 표시하면 다음과 같습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/576f11c2-aaf6-47d6-affe-23342f088322)

위에서 적용했던 0.9를 β로 변경해서 나타내면 Vt = βVt-1 + (1-β)θt가 됩니다.
만약에 β가 0.9가 되는 경우라면 이전 10일 동안의 평균기온과 비슷하다고 생각하면됩니다.
만약에 β가 0.98이라면 지난 50일의 평균기온과 비슷하다고 보면 됩니다.
만약에 β가 0.5이리면 2일간의 평균기온이기 때문에 아래 노란색 그래프처럼 noisy한 결과를 얻게 될 것이고 이상치게 취약하게 되지만 기온이 더 빨리 변하는 것을 반영시켜줍니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/3181a889-188f-4acb-9a6f-8249aa79995f)

### Understanding Exponentially Weighted Averages
 Vt = βVt-1 + (1-β)θt 방정식이 있고 아래와 같이 감소하는 형태라면 V100의 의미를 살펴보기 위해서 쭉 나열해보면 일별 기온을 기하급수적으로 감소하는 함수에 곱해서 더하는 것입니다.
 ![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/4a78d66f-5ab2-47c8-91a5-422d9d037972)

이런 점 때문에 지수가중평균이라고 합니다. 여기서 0.9의 10승을 하게 되면 그 값이 0.35정도가 되는데, 이 값은 대략 1/e​ 입니다. 다시 말해서, 지수 그래프가 3분의1 정도로 감소하는데 약 10일 정도가 소요되고, β가 0.9인 경우에 직전 10일간의 기온만 집중하여 지수 가중 평균을 구하는 것과 같습니다.
반면 β의 값이 0.98이라고 한다면 0.98의 50승이면 대략 1/e​와 비슷한 값이 되고, 이 값은 50일간의 대략적인 평균치라고 보면 됩니다.
​마지막으로 이것을 구현하는 방법은 하나의 Vθ의 값을 가지고 갱신하면서 지수이동편균을 구할 수 있습니다. 지수이동평균을 구하는 것의 장점은 아주 적은 양의 메모리가 이용된다는 것입니다.

### Bias Correction in exponentially weighted averages
지수이동편균을 구현하는 법에 대해서 배웠는데 여기에 bias를 보정하는 방법이있습니다.
만약에 β의 값이 0.98이라면 실제로는 보라색 그림처럼 그려질 것입니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/337cbcbd-dff9-4640-9e4a-d0e3fa4444a8)

가중이동평균을 구할 때 화씨가 40도인 경우에 v0은 0으로 초기화를 하고 시작하기 때문에 0.98x0으로 의미없는 값이 됩니다. 그래서 둘째날의 기온은 v1은 0.02*40으로 8이 되기 때문에 너무 작은 값이 도출됩니다. 초반에는 좋은 값이 아니기 때문에 이것을 보정하기 위해서 bias보정 방법을 사용합니다. 이 때 vt 대신에 vt/(1-βt)를 사용합니다.
이렇게 초기 지수이동평균값을 1-βt로 보정해주고 t값이 충분히 커지면 βt는 0으로 수렴하게 되고 bias보정 영향은 사라지게 됩니다.

### Gradient Descent with Momentum
Momentum 또는 Gradient Descent with momentum이라고 하는 알고리즘이 있습니다. 이 알고리즘은 일반적인 기울기 하강 알고리즘보다 더 빨리 작동합니다. 기본아이디어는 기울기의 지수가중평균을 산출하는 것입니다. 그 다음에 이 기울기를 이용해서 가중을 업데이트 하는 것입니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/e5fc41bb-4fc6-42c0-8876-98003df5c5b7)

이러한 곡선을 가진 비용함수를 최적화 한다고 할 때, 빨간점은 최소점의 위치를 나타냅니다. 기울기 하강이 여러차례 일어나 천천히 최소값을 향해 접근하게 될 것입니다. 이렇게 왔다갔다하는 변동폭이 기울기 하강을 느리게 하고 더 큰 학습률을 쓰지 않게 합니다. 만약 더 큰 학습률을 쓰게 한다면 발산하거나 갈라지게 할 수 있습니다. 따라서 진동이 너무 커지는 것을 방지해야 하기 때문에 너무 크지 않는 학습률을 사용해야합니다. 진동하는 폭을 감소시키기 위해서 우리는 세로축에서는 slower learning, 가로축에 대해서는 faster learning을 원할 것입니다. 이것은 Gradient Descent with momentum을 통해서 감소시킬 수 있습니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/4650c8a8-f474-4112-91b2-2982e4f74051)

구하는 과정은 현재의 mini-batch에서 dW와 dB를 구합니다. 그 다음에는 dW와 dB에 대해서 EMA를 구하고, 이렇게 구한 EMA를 가지고 W와 b 파라미터를 업데이트합니다. 이 과정을 통해서 경사하강법 더욱 부드럽게 해주게 됩니다. 결과적으로 세로축의 변동 평균은 거의 0이되고, 가로축의 변동 평균이 꽤 큰 값이 되어서, 아래와 같이 빨간색 그래프처럼 최소값을 향해서 접근하게 됩니다.알고리즘에서 dW와 db는 가속도의 역할을 하고, VdW,Vdb는 속도의 역할을 하게 되는 것입니다.

### RMSprop
Momentum을 사용하면 경사하강의 속도를 높일 수 있습니다. 또 다른 RMSprop이라는 알고리즘이 있습니다. Root Mean Square prop의 약자로 경사하강의 속도를 높일 수도 있습니다.
세로축을 파라미터 b라고 하고 가로축을 파라미터 w라고 할 때, 우리는 세로축은 더 느리고 가로축은 더 빠르게 학습하기를 원합니다.
RMSprop 알고리즘은 현재 mini-batch에 대해서 dW, db를 구하고, 이번에는 SdW 표기를 사용해서 구하게 되는데 dW와 db를 제곱해서 SdW와 Sdb를 구하면 됩니다. 이것은 지수가중이동평균을 유지하 것을 의미합니다. 가로방향의 변동은 증가시키고 세로방향의 변동은 늦추고 싶기 때문에 SdW는 비교적 작은 값이 되어야 하고 Sdb는 비교적 큰 값이 되어야 합니다.
실제 미분항을 보면, 가로방향에서보다 세로방향으로 더 큰 것을 볼 수 있고, 기울기는 b방향으로 더 크게 됩니다. 

이것이 RMSprop이며 momentum과 비슷하게 경사하강을 진행할 때의 진동을 감소시키는 효과가 있습니다. 이렇게 진동을 감소시키면, 더 큰 learning rate α를 사용할 수 있고, 학습 알고리즘의 속도를 증가시킬 수 있습다.

### Adam optimization algorithm
Adam은 adptive moment estimation의 약자로  Momentum과 RMSprop를 합치는 것이라고 생각하면 됩니다.
Adam을 구하려면 V_dW=0, SdW=0으로 초기화하면 되며 마찬가지로 V_db, S_db =0입니다. t를 반복하면서 미니배치를 사용하여 dw와 db를 계산합니다. 그리고 모멘텀을 진행하고 RMSprop도 진행하며 업데이트 해줍니다. 추가적으로 Adam에서는 Bias Correction도 같이 진행하게 됩니다. 그렇게 해서 구해진 VcorrecteddW, Vcorrecteddb, ScorrecteddW, Scorrecteddb를 가지고 W,b파라미터를 업데이트 하면 됩니다.
이 알고리즘에는 다양한 하이퍼파라미터가 있습니다. 그 중 learning rate α는 매우 중요하고 β1의 기본값은 0.9이고 β2의 값은 0.999를 권장하고, ϵ의값은 10-8을 권장하지만 성능에 거의 영향이 없기 때문에 바꾸지않아도 됩니다.

### Learning Rate Decay
알고리즘을 더 빠르게 학습시킬 수 있는 방법 중 하나는 learing rate를 시간의 지남에 따라 학습률을 천천히 줄이는 것을 의미합니다. mini-batch GD를 사용한다고 가정해보면, 우리는 학습을 반복하면서 최소값으로 향하는 경향이 있지만, 정확하게 최소값으로 수렴하지는 않을 것입니다.
![image](https://github.com/ellieso/coursera-deep-learning-specialization/assets/83899219/124ddda4-5f20-4392-adc8-e9cac4529c75)

위와 같이 최소값 주변을 멤돌면서 절대로 수렴하지 않을 것입니다. 이것은 learning rate를 어떤 값으로 고정시켰고, noise가 존재하기 때문입니다. 만약에 learning rate를 천천히 감소시킨다면 처음에는 빠르게 학습할 것인데 점점 줄어들어 매우 좁은 범위를 왔다갔다 할 것입니다.
Learning Rate Decay의 구현은 1 / (1+ decay rate * epoch-num) * α0 입니다. epoch는 각 mini-batch set에 대해서 진행하는 것을 의미하고, α0는 초기 learning rate입니다.
위 수식 외에도 다른 방법이 있는데 (0.95 * epoch-num * α0) - exponentially decay / (k/ 루트 epoch-num)* α0 / discrete staircase /  수동으로 decay 입니다.
