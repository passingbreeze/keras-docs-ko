# Keras: The Python Deep Learning library

<img src='https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png' style='max-width: 600px; width: 90%;' />



## You have just found Keras.

Keras는 Python으로 작성된 고수준의 신경망 API로, [TensorFlow](https://github.com/tensorflow/tensorflow)나 [CNTK](https://github.com/Microsoft/cntk) 혹은 [Theano](https://github.com/Theano/Theano)와 같은 기계학습 라이브러리를 기반으로 하여 동작합니다. Keras는 실험을 빠르게 진행할 수 있도록 만드는데에 중점을 두고 개발되었습니다. *좋은 연구의 관건은 어떤 생각으로부터 결과를 도출하기 까지의 지연을 최소화하는 것에 있습니다.*

다음의 상황에서 deep learning 라이브러리가 필요하다면 Keras를 활용하십시오.

 - 간단한 프로토타입을 빠르게 만들어야할 때 (사용하기 쉽고 모듈식이며 확장성이 좋은 Keras의 특징을 통해)
 - 합성곱(Convolutional) 신경망과 순환(Recurrent) 신경망을 합쳐야할 뿐만 아니라 두 신경망 모두를 지원해야할때
 - CPU와 GPU 환경에서 매끄럽게 동작해야할 때

[Keras.io](https://keras.io)에 있는 문서를 읽으십시오.

Keras는 __Python 2.7-3.6__ 에서 원활하게 작동합니다.

------------------

## 아래는 Keras를 사용하면서 반드시 지켜야하는 사항들입니다.

- __사용자에게 친숙하도록.__ Keras는 지속적이면서도 단순한 API를 제공하고, 일반적인 경우에 요구되는 사용자의 작업횟수를 최소하하며, 사용자에 의해 발생하는 오류는 명확하고 정확한 피드백을 제공합니다. 이러한 특징들은 사용자들이 Keras를 사용함에 있어서 눈에 띄는 부하를 줄여주는데 최선의 방법을 제공할 것입니다.

- __모듈성(Modularity)을 띄도록.__ 모델은 가능한한 제한을 많이 하지 않는 조건 하에서 서로 연결될 수 있는, 완전히 조정된 모듈의 나열이나
그래프(정점(Vertex)들의 집합(V)과 서로 다른 정점들을 연결하는 간선(Edge)들의 집합으로 구성된 구조)로 알려져있습니다. 실제로 신경망 층들, 지도학습에서 예측된 결과 값과 실제 결과 값의 차이를 표현해주는 비용 함수들, 신경망 층들을 학습시키는 옵티마이저들, 신경망을 초기화하는 방식들, 활성 함수들, 정규화는 새로운 모델들을 만들때 같이 결합해서 사용할 수 있는 독립적인 모듈들입니다.

- __쉽게 넓혀나갈 수 있도록.__ 새로운 모듈들은 간단하게 추가할 수 있고(새로운 클래스와 함수로써) 기존에 있던 모듈들은 충분한 예시를 제공해주고 있습니다. 새로운 모듈들을 쉽게 추가할 수 있도록 한 점은 어떤 분야에서건 구현하고 싶은 모듈들을 만들 수 있도록 허용해준다는 것이고 이것은 좀더 심화된 연구에서 Keras를 사용하기에 더욱 적합하도록 만들어줍니다.

- __Python으로 짜도록.__ 설정 파일들에는 선언적 형식으로된 별도의 모델들이 없습니다. 모델들은 간결하면서도 수정하기 쉽고 확장하기도 쉬운 Python 코드로 적혀져야 합니다.


------------------


## 시작해봅시다 : Keras로 30초만에 신경망 모델 만들어보기.

Keras의 가장 핵심이 되는 자료구조는 층을 구성하는 방식 중에 하나인 __model__입니다. 가장 단순한 형태의 모델은 일직선 방향으로 층들이 쌓아 올려진 [`Sequential`](https://keras.io/getting-started/sequential-model-guide) 모델입니다. 좀더 복잡한 형태의 모델을 구성하기 위해서는 임의의 그래프 구조를 층에 만들 수 있도록 하는 [Keras 함수형 API](https://keras.io/getting-started/functional-api-guide)를 반드시 사용해야합니다.

여기 `Sequential` 모델이 있습니다.

```python
from keras.models import Sequential

model = Sequential()
```

`.add()` 를 사용해서 층을 쉽게 쌓을 수 있습니다.:

```python
from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
```

여러분들의 모델이 잘 만들어진 것 같다면, `.compile()`을 사용해 학습 과정을 조정해봅시다.:

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

If you need to, you can further configure your optimizer. A core principle of Keras is to make things reasonably simple, while allowing the user to be fully in control when they need to (the ultimate control being the easy extensibility of the source code).

```python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
```

You can now iterate on your training data in batches:

```python
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

Alternatively, you can feed batches to your model manually:

```python
model.train_on_batch(x_batch, y_batch)
```

Evaluate your performance in one line:

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

Or generate predictions on new data:

```python
classes = model.predict(x_test, batch_size=128)
```

Building a question answering system, an image classification model, a Neural Turing Machine, or any other model is just as fast. The ideas behind deep learning are simple, so why should their implementation be painful?

For a more in-depth tutorial about Keras, you can check out:

- [Getting started with the Sequential model](https://keras.io/getting-started/sequential-model-guide)
- [Getting started with the functional API](https://keras.io/getting-started/functional-api-guide)

In the [examples folder](https://github.com/keras-team/keras/tree/master/examples) of the repository, you will find more advanced models: question-answering with memory networks, text generation with stacked LSTMs, etc.


------------------


## Installation

Before installing Keras, please install one of its backend engines: TensorFlow, Theano, or CNTK. We recommend the TensorFlow backend.

- [TensorFlow installation instructions](https://www.tensorflow.org/install/).
- [Theano installation instructions](http://deeplearning.net/software/theano/install.html#install).
- [CNTK installation instructions](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine).

You may also consider installing the following **optional dependencies**:

- [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/) (recommended if you plan on running Keras on GPU).
- HDF5 and [h5py](http://docs.h5py.org/en/latest/build.html) (required if you plan on saving Keras models to disk).
- [graphviz](https://graphviz.gitlab.io/download/) and [pydot](https://github.com/erocarrera/pydot) (used by [visualization utilities](https://keras.io/visualization/) to plot model graphs).

Then, you can install Keras itself. There are two ways to install Keras:

- **Install Keras from PyPI (recommended):**

```sh
sudo pip install keras
```

If you are using a virtualenv, you may want to avoid using sudo:

```sh
pip install keras
```

- **Alternatively: install Keras from the GitHub source:**

First, clone Keras using `git`:

```sh
git clone https://github.com/keras-team/keras.git
```

 Then, `cd` to the Keras folder and run the install command:
```sh
cd keras
sudo python setup.py install
```

------------------


## Configuring your Keras backend

By default, Keras will use TensorFlow as its tensor manipulation library. [Follow these instructions](https://keras.io/backend/) to configure the Keras backend.

------------------


## Support

You can ask questions and join the development discussion:

- On the [Keras Google group](https://groups.google.com/forum/#!forum/keras-users).
- On the [Keras Slack channel](https://kerasteam.slack.com). Use [this link](https://keras-slack-autojoin.herokuapp.com/) to request an invitation to the channel.

You can also post **bug reports and feature requests** (only) in [GitHub issues](https://github.com/keras-team/keras/issues). Make sure to read [our guidelines](https://github.com/keras-team/keras/blob/master/CONTRIBUTING.md) first.


------------------


## Why this name, Keras?

Keras (κέρας) means _horn_ in Greek. It is a reference to a literary image from ancient Greek and Latin literature, first found in the _Odyssey_, where dream spirits (_Oneiroi_, singular _Oneiros_) are divided between those who deceive men with false visions, who arrive to Earth through a gate of ivory, and those who announce a future that will come to pass, who arrive through a gate of horn. It's a play on the words κέρας (horn) / κραίνω (fulfill), and ἐλέφας (ivory) / ἐλεφαίρομαι (deceive).

Keras was initially developed as part of the research effort of project ONEIROS (Open-ended Neuro-Electronic Intelligent Robot Operating System).

>_"Oneiroi are beyond our unravelling --who can be sure what tale they tell? Not all that men look for comes to pass. Two gates there are that give passage to fleeting Oneiroi; one is made of horn, one of ivory. The Oneiroi that pass through sawn ivory are deceitful, bearing a message that will not be fulfilled; those that come out through polished horn have truth behind them, to be accomplished for men who see them."_ Homer, Odyssey 19. 562 ff (Shewring translation).

------------------
