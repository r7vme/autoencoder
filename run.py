#!/usr/bin/env python

"""
Run autoencoder model.
"""
import numpy as np
import tensorflow as tf
from PIL import Image


def save_img(d, path="autoencoder_sample.jpg"):
    arr = (d[0] + 1.0) * 127.5
    arr = arr.astype(np.uint8)
    im = Image.fromarray(arr)
    im.save(path)

if __name__ == "__main__":
    from model import get_model

    with tf.Session() as sess:
        _, _, _, _, _, [G, _, E] = get_model(sess=sess, name="autoencoder", batch_size=1)
        G.load_weights("outputs/results_autoencoder/G_weights.keras")
        E.load_weights("outputs/results_autoencoder/E_weights.keras")

        im = Image.open("images/img1545683165.5584788.jpg")
        x = np.asarray(im).reshape((-1, 80, 160, 3))
        z = E.predict(x, batch_size=1)[0]
        out = G.predict(z, batch_size=1)
        save_img(out)
