# Advanced Analytics II (BA885)

This repository was originally created to host the course materials for the Advanced Analytics II course first offered in Spring 2022 at Boston University's [Questrom School of Business](https://www.bu.edu/questrom/).

The notebooks presented here are sporadically updated with better examples to include more advanced techniques and improve readability. 

For instance, the recently updated [transfer learning notebook](https://github.com/ndoroud/BA885/tree/master/05_Transfer_Learning.ipynb) offers a neat example of transfer learning by re-purposing the pre-trained [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) model, part of [TensorFlow Model Garden] (https://github.com/tensorflow/models), to recognize speech commands using the dataset published by Peter Warden in conjunction with the paper [1804.03209](https://arxiv.org/abs/1804.03209). The notebook introduces applications of 'tensorflow.signal' to audio data *with channel support*, which is somewhat lacking in the commonly used [tensorflow_io.audio](https://www.tensorflow.org/io/api_docs/python/tfio/audio).

In these notebooks I have assumed familiarity with the basics of Deep Learning using Keras/TensorFlow, a delve deeper into various aspects
of Deep Learning using a very hands-on approach. The original notebooks for BA885 course can be found in the [v1_Spring_2022](https://github.com/ndoroud/BA885/tree/master/v1_spring_2022) folder.

Furthermore, I have taken a modular approach to model building with Keras and TensorFlow and the notebooks are thus desiged to
familiarize the audience with piecing together blocks/modules to build more complex neural network architechtures.

Some of the topics covered are
- Transfer Learning
- Ensembling
- CNNs and RNNs
- Transformers
- Generative models
- Unsupervised Learning
- Reinforcement Learning
