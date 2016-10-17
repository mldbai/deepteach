# DeepTeach - the Interactive Deep Image Classifier Builder

DeepTeach is an MLDB.ai plugin that allows a user to teach a machine learning model what types of 
images he’s looking for through an iterative process. It's a great example of human augmentation, where 
machine learning is used to make humans more efficient.

The plugin uses the Inception-v3 model, a deep convolutional neural network, as its feature generator. It 
then uses the user's input to train a bagged boosted decision tree in order to learn the type of image the
user is looking for. It's a combination of active learning, deep learning, transfer learning and similarity search.

Some links:

- [DeepTeach Youtube Demo](https://youtu.be/7hZ3X37Qwc4)
- [DeepTeach Blog Post](http://blog.mldb.ai/blog/posts/2016/10/deepteach/)
- [KDNuggets guest blog post](http://www.kdnuggets.com/2016/10/mldb-machine-learning-database.html)

Try *DeepTeach* for free! Just create a [free MLDB.ai account](https://mldb.ai/#signup) 
to launch an instance and run the 
[Transfer Learning on Images with Tensorflow demo](https://docs.mldb.ai/ipy/notebooks/_demos/_latest/Transfer%20Learning%20with%20Tensorflow.html) from within your MLDB instance.

### Installing DeepTeach

One way is to the bottom of the [Transfer Learning on Images with Tensorflow demo](https://docs.mldb.ai/ipy/notebooks/_demos/_latest/Transfer%20Learning%20with%20Tensorflow.html) notebook from a running instance of MLDB.

Alternatively, from a notebook running on MLDB, run the following:

```python
from pymldb import Connection
mldb = Connection()

mldb.put("/v1/plugins/deepteach", {
    "type": "python",
    "params": {
        "address": "git://github.com/mldbai/deepteach"
    }
})
```

You can then browse to `https://<host:port>/v1/plugins/deepteach/routes/static/index.html` to access the UI.
