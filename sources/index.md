# Keras: The Python Deep Learning library

<img src='https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png' style='max-width: 600px; width: 90%;' />



## You have just found Keras.

Keras는 Python으로 작성된 고수준의 신경망 API로, [TensorFlow](https://github.com/tensorflow/tensorflow)나 [CNTK](https://github.com/Microsoft/cntk) 혹은 [Theano](https://github.com/Theano/Theano)와 같은 기계학습 라이브러리를 기반으로 하여 동작합니다. Keras는 실험을 빠르게 진행할 수 있도록 만드는데에 중점을 두고 개발되었습니다. *좋은 연구의 관건은 어떤 생각으로부터 결과를 도출하기 까지의 지연을 최소화하는 것에 있습니다.*

다음의 상황에서 deep learning 라이브러리가 필요하다면 Keras를 활용하십시오.

 - 빠르게 간단한 프로토타입을 만들어야할 때 (사용하기 쉽고 모듈식이며 확장성이 좋은 Keras의 특징을 통해)
 - 합성곱 신경망과 순환 신경망을 합쳐야할 뿐만 아니라 두 신경망 모두를 지원해야할때
 - CPU와 GPU 환경에서 매끄럽게 동작해야할 때

[Keras.io](https://keras.io)에 있는 문서를 읽으십시오.

Keras는 __Python 2.7-3.6__에서 원활하게 작동합니다.

------------------

## Guiding principles

- __User friendliness.__ Keras is an API designed for human beings, not machines. It puts user experience front and center. Keras follows best practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear and actionable feedback upon user error.

- __Modularity.__ A model is understood as a sequence or a graph of standalone, fully-configurable modules that can be plugged together with as few restrictions as possible. In particular, neural layers, cost functions, optimizers, initialization schemes, activation functions, regularization schemes are all standalone modules that you can combine to create new models.

- __Easy extensibility.__ New modules are simple to add (as new classes and functions), and existing modules provide ample examples. To be able to easily create new modules allows for total expressiveness, making Keras suitable for advanced research.

- __Work with Python__. No separate models configuration files in a declarative format. Models are described in Python code, which is compact, easier to debug, and allows for ease of extensibility.


------------------


## Getting started: 30 seconds to Keras

The core data structure of Keras is a __model__, a way to organize layers. The simplest type of model is the [`Sequential`](https://keras.io/getting-started/sequential-model-guide) model, a linear stack of layers. For more complex architectures, you should use the [Keras functional API](https://keras.io/getting-started/functional-api-guide), which allows to build arbitrary graphs of layers.

Here is the `Sequential` model:

```python
from keras.models import Sequential

model = Sequential()
```

Stacking layers is as easy as `.add()`:

```python
from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
```

Once your model looks good, configure its learning process with `.compile()`:

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
