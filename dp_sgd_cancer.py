import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, initializers
from tensorflow.keras.layers import Dense, Flatten
import os

# from dp_compression.mechanisms import MVUMechanism
import erm, optm, rqm, genm
import argparse


tf.config.run_functions_eagerly(True)

from sklearn.datasets import load_breast_cancer 
data = load_breast_cancer()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def dp_sgd_train(pri_eps, mechanism, clip_norm, batch_size, **kw):

    batch_arr = []
    c = clip_norm
    m = 4

    def quant(grad, mechanism):

        shape = np.shape(grad.numpy())
        res = []

        # Only clipping the gradient, without privacy guarantee
        if mechanism == "Clip":
            return grad

        # if mechanism == "MVU":
        #     res = ((grad.numpy() + clip_norm) / (2 * clip_norm)).flatten()
        #     for time in range(100):
        #         try:
        #             mechanism = MVUMechanism(budget=2, epsilon=pri_eps, input_bits=2, method="trust-region")
        #             res = (mechanism.decode(mechanism.privatize(res)) * 2 * clip_norm) - clip_norm
        #             break
        #         except:
        #             print("Error", time)
        #             continue
        
        if mechanism == 'RQM':
            grad = grad.numpy().flatten()
            rqm_delta, rqm_q = rqm.opt_par(c, m, pri_eps)
            rqm_bins = np.linspace(-c-rqm_delta, c+rqm_delta, 4)
            for id in range(len(grad)):
                rqm_sample_pr = rqm.calc_sample_pr(rqm_bins, grad[id], rqm_q)
                rqm_bins_pr = genm.calc_bins_pr(rqm_bins, rqm_sample_pr, grad[id])
                res.append(np.random.choice(rqm_bins, p=rqm_bins_pr))

        if mechanism == 'OPTM':
            # optm_bins, optm_q_arr = optm.opt_par(c, m, pri_eps)

            # can use customized bin values
            optm_bins = [-2.2, -0.4, 0.4, 2.2]
            _, optm_q_arr = optm.opt_par(c, m, pri_eps, bins=optm_bins)

            grad = grad.numpy().flatten()
            for id in range(len(grad)):
                nuj = -1
                for bid in range(m-1):
                    if optm_bins[bid] <= grad[id] < optm_bins[bid+1]:
                        nuj = bid
                assert nuj != -1, f"Wrong j for OPTM"
                
                optm_bins_pr = genm.calc_bins_pr(optm_bins, optm_q_arr[nuj], grad[id])
                res.append(np.random.choice(optm_bins, p=optm_bins_pr))

        res = tf.cast(tf.reshape(res, shape), dtype=tf.float32)
        return res

    @tf.custom_gradient
    def custom_op(y):
        def custom_grad(upstream):
            upstream = tf.clip_by_value(upstream, -clip_norm, clip_norm)
            upstream = quant(upstream, mechanism)
            return upstream
        return y, custom_grad

    class CustomCallback(keras.callbacks.Callback):

        def on_train_batch_end(self, batch, logs=None):
            batch_arr.append(logs["acc"])

    class CustomLayer(tf.keras.layers.Layer):
        def __init__(self):
            super(CustomLayer, self).__init__()

        def call(self, inputs):
            return custom_op(inputs)
        

    class MyModel(Model):
        def __init__(self):
            super().__init__()
            self.flatten = Flatten()
            self.layer1 = Dense(2, kernel_initializer=initializers.RandomNormal(seed=None))
            self.layer2 = CustomLayer()

        def call(self, x):
            x = self.flatten(x)
            x = self.layer1(x)
            return self.layer2(x)

    model = MyModel()

    model.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="acc")
        ]
    )
                
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=1,
        verbose=2,
        # validation_split=0.25,
        callbacks=[CustomCallback()],
    )

    # return history.history['val_acc']
    return batch_arr


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=float, default=0.1, help='clipping range of input')
    parser.add_argument('--eps', type=float, default=1, help='privacy budget parameter epsilon')

    args = parser.parse_args()

    batch_size = 8
    clip_norm = args.c
    num_runs = 30

    c = clip_norm
    m = 4

    pri_eps = args.eps
    pri = math.exp(pri_eps)

    dir = f"cancer_eps{pri_eps}_clip{clip_norm}_b{batch_size}_sr_train/"
    if os.path.exists(dir) == False:
        os.makedirs(dir)

    clip_arr = []
    optm_arr = []
    mvu_arr = []
    rqm_arr = []

    for mechanism in ['OPTM', 'RQM']:
        for i in range(num_runs):
            res = dp_sgd_train(pri_eps, mechanism, clip_norm, batch_size)
            if mechanism == 'Clip':
                clip_arr = np.append(clip_arr, np.array([res]), axis=0)
            if mechanism == 'OPTM':
                optm_arr.append(res)
            # if mechanism == 'MVU':
            #     mvu_arr = np.append(mvu_arr, np.array([res]), axis=0)
            if mechanism == 'RQM':
                rqm_arr = np.append(rqm_arr, np.array([res]), axis=0)
          

    np.save(dir+'clip.npy', np.array(clip_arr))
    np.save(dir+'optm.npy', np.array(optm_arr))
    # np.save(dir+'mvu.npy', np.array(mvu_arr))
    np.save(dir+'rqm.npy', np.array(rqm_arr))

    mvu_mean_arr = []
    mvu_std_arr = []
    optm_mean_arr = []
    optm_std_arr = []
    rqm_mean_arr = []
    rqm_std_arr = []
    erm_mean_arr = []
    erm_std_arr = []

    clip_mean_arr = []
    clip_std_arr = []

    tmp_mvu_arr = []
    tmp_optm_arr = []
    tmp_rqm_arr = []
    tmp_erm_arr = []
    tmp_clip_arr = []

    for id in range(len(optm_arr[0])):

        tmp_clip_arr = [arr[id] for arr in clip_arr]
        clip_mean_arr.append(np.mean(tmp_clip_arr))
        clip_std_arr.append(np.std(tmp_clip_arr))

        # tmp_mvu_arr = [arr[id] for arr in mvu_arr]
        # mvu_mean_arr.append(np.mean(tmp_mvu_arr))
        # mvu_std_arr.append(np.std(tmp_mvu_arr))

        tmp_optm_arr = [arr[id] for arr in optm_arr]
        optm_mean_arr.append(np.mean(tmp_optm_arr))
        optm_std_arr.append(np.std(tmp_optm_arr))

        tmp_rqm_arr = [arr[id] for arr in rqm_arr]
        rqm_mean_arr.append(np.mean(tmp_rqm_arr))
        rqm_std_arr.append(np.std(tmp_rqm_arr))


    x = np.array(range(1, len(optm_arr[0])+1))
    plt.errorbar(x, clip_mean_arr, yerr=clip_std_arr, label="Without privacy")
    # plt.errorbar(x, mvu_mean_arr, yerr=mvu_std_arr, label="MVU")
    plt.errorbar(x, optm_mean_arr, yerr=optm_std_arr, label="OPTM")
    plt.errorbar(x, rqm_mean_arr, yerr=rqm_std_arr, label="RQM")

    plt.legend()
    plt.show()





