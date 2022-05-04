# %%
import matplotlib.pyplot as plt
from transformer import *
import tensorflow as tf
# %%
small_transformer = transformer(
    vocab_size = 9000,
    num_layers = 4,
    dff = 100,
    value_depth = 16,
    d_model = 128,
    num_heads = 4,
    dropout = 0.3,
    name="small_transformer")

tf.keras.utils.plot_model(
    small_transformer, show_shapes=True)
# %%
small_transformer.summary()
# %%
