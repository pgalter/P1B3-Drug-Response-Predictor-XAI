#! /usr/bin/env python
#! /usr/bin/env python

"""multilayer perceptron for drug response problem"""

from __future__ import division, print_function

import argparse
import csv
import logging
import sys

import numpy as np

from keras import backend as K
from keras import metrics
from keras.models import Sequential
from keras.layers import (
    Activation, BatchNormalization, Dense, Dropout,
    LocallyConnected1D, Conv1D, MaxPooling1D, Flatten,
    Conv2D, LocallyConnected2D
)
from keras.callbacks import Callback, ModelCheckpoint, ProgbarLogger

# non-interactive plotting
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import p1b3 as benchmark
import candle


def initialize_parameters(default_model='p1b3_default_model.txt'):
    """
    load and finalize parameters for the benchmark

    returns:
        dict: all parameters used to run the model
    """
    # build benchmark object
    p1b3Bmk = benchmark.BenchmarkP1B3(
        benchmark.file_path,
        default_model,
        'keras',
        prog='p1b3_baseline',
        desc='multi-task dnn for clinical data (pilot 3 benchmark 1)'
    )

    # finalize parameters
    gParameters = candle.finalize_parameters(p1b3Bmk)

    return gParameters


def str2lst(string_val):
    """
    convert a space-separated string into a list of ints

    example: "64 32 16" -> [64, 32, 16]
    """
    return [int(x) for x in string_val.split(' ')]


def evaluate_keras_metric(y_true, y_pred, metric):
    """
    evaluate a keras metric manually

    args:
        y_true: true labels
        y_pred: predicted values
        metric: metric name (string)

    returns:
        float: metric value
    """
    objective_function = metrics.get(metric)
    objective = objective_function(y_true, y_pred)
    return K.eval(objective)


def evaluate_model(model, generator, steps, metric, category_cutoffs=[0.]):
    """
    run model on generator and compute loss + accuracy

    returns:
        loss, acc, and raw predictions
    """
    y_true, y_pred = None, None
    count = 0

    # loop through batches
    while count < steps:
        x_batch, y_batch = next(generator)

        y_batch_pred = model.predict_on_batch(x_batch).ravel()

        # build full arrays
        y_true = np.concatenate((y_true, y_batch)) if y_true is not None else y_batch
        y_pred = np.concatenate((y_pred, y_batch_pred)) if y_pred is not None else y_batch_pred

        count += 1

    # compute loss
    loss = evaluate_keras_metric(
        y_true.astype(np.float32),
        y_pred.astype(np.float32),
        metric
    )

    # convert to classes
    y_true_class = np.digitize(y_true, category_cutoffs)
    y_pred_class = np.digitize(y_pred, category_cutoffs)

    # compute accuracy (cast to float for compatibility)
    acc = evaluate_keras_metric(
        y_true_class.astype(np.float32),
        y_pred_class.astype(np.float32),
        'binary_accuracy'
    )

    return loss, acc, y_true, y_pred, y_true_class, y_pred_class


def plot_error(y_true, y_pred, batch, file_ext, file_pre='output_dir', subsample=1000):
    """
    plot prediction errors and save figures

    only runs every 10 batches to avoid too many plots
    """
    if batch % 10:
        return

    total = len(y_true)

    # subsample if dataset is large
    if subsample and subsample < total:
        idx = np.random.choice(total, size=subsample, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    # convert to percentage
    y_true = y_true * 100
    y_pred = y_pred * 100

    diffs = y_pred - y_true

    bins = np.linspace(-200, 200, 100)

    # baseline random comparison
    if batch == 0:
        y_shuf = np.random.permutation(y_true)
        plt.hist(y_shuf - y_true, bins, alpha=0.5, label='random')

    plt.hist(diffs, bins, alpha=0.3, label=f'epoch {batch+1}')
    plt.title("error histogram (%)")
    plt.legend(loc='upper right')

    plt.savefig(f"{file_pre}.histogram{file_ext}.b{batch}.png")
    plt.close()

    # scatter plot
    fig, ax = plt.subplots()
    plt.grid('on')

    ax.scatter(y_true, y_pred, s=10)
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()], 'k--', lw=2)

    ax.set_xlabel('measured')
    ax.set_ylabel('predicted')

    plt.savefig(f"{file_pre}.diff{file_ext}.b{batch}.png")
    plt.close()


class MyLossHistory(Callback):
    """
    custom callback to track validation and test performance
    """

    def __init__(self, progbar, val_gen, test_gen, val_steps, test_steps,
                 metric, category_cutoffs=[0.], ext='', pre='save'):
        super(MyLossHistory, self).__init__()

        self.progbar = progbar
        self.val_gen = val_gen
        self.test_gen = test_gen
        self.val_steps = val_steps
        self.test_steps = test_steps
        self.metric = metric
        self.category_cutoffs = category_cutoffs
        self.pre = pre
        self.ext = ext

    def on_train_begin(self, logs=None):
        # track best values
        self.best_val_loss = np.Inf
        self.best_val_acc = -np.Inf

    def on_epoch_end(self, batch, logs=None):
        logs = logs or {}

        # evaluate on val and test
        val_loss, val_acc, y_true, y_pred, *_ = evaluate_model(
            self.model, self.val_gen, self.val_steps,
            self.metric, self.category_cutoffs
        )

        test_loss, test_acc, *_ = evaluate_model(
            self.model, self.test_gen, self.test_steps,
            self.metric, self.category_cutoffs
        )

        # update progress bar
        self.progbar.append_extra_log_values([
            ('val_acc', val_acc),
            ('test_loss', test_loss),
            ('test_acc', test_acc)
        ])

        # plot only if val improves
        if float(logs.get('val_loss', 0)) < self.best_val_loss:
            plot_error(y_true, y_pred, batch, self.ext, self.pre)

        self.best_val_loss = min(float(logs.get('val_loss', 0)), self.best_val_loss)
        self.best_val_acc = max(float(logs.get('val_acc', 0)), self.best_val_acc)


class MyProgbarLogger(ProgbarLogger):
    """
    custom progress logger with extra metrics
    """

    def __init__(self, samples):
        super(MyProgbarLogger, self).__init__(count_mode='samples')
        self.samples = samples

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        self.verbose = 1
        self.extra_log_values = []
        self.params['samples'] = self.samples

    def append_extra_log_values(self, tuples):
        # store extra metrics to print later
        for k, v in tuples:
            self.extra_log_values.append((k, v))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        for k, v in self.extra_log_values:
            self.log_values.append((k, v))

        if self.verbose:
            self.progbar.update(self.seen, self.log_values)


def add_conv_layer(model, layer_params, input_dim=None, locally_connected=False):
    """
    add a convolutional or locally connected layer

    layer_params:
        [filters, kernel, stride] for 1d
        [filters, k1, k2, s1, s2] for 2d
    """
    # 1d conv
    if len(layer_params) == 3:
        filters, kernel, stride = layer_params

        if locally_connected:
            layer = LocallyConnected1D(filters, kernel, strides=stride,
                                      input_shape=(input_dim, 1) if input_dim else None)
        else:
            layer = Conv1D(filters, kernel, strides=stride,
                           input_shape=(input_dim, 1) if input_dim else None)

    # 2d conv
    elif len(layer_params) == 5:
        filters = layer_params[0]
        kernel = (layer_params[1], layer_params[2])
        stride = (layer_params[3], layer_params[4])

        if locally_connected:
            layer = LocallyConnected2D(filters, kernel, strides=stride,
                                      input_shape=(input_dim, 1) if input_dim else None)
        else:
            layer = Conv2D(filters, kernel, strides=stride,
                           input_shape=(input_dim, 1) if input_dim else None)

    model.add(layer)
    return model


def run(gParameters):
    """
    run the full training pipeline

    this includes:
    - preparing parameters
    - building the model
    - training
    - evaluation
    - simple xai step at the end

    args:
        gParameters (dict): config values for the run
    """

    # make sure dense layers are in list form
    if 'dense' in gParameters:
        dval = gParameters['dense']
        if type(dval) != list:
            gParameters['dense'] = list(dval)
        print(gParameters['dense'])

    # reshape conv params into groups
    if 'conv' in gParameters:
        flat = gParameters['conv']
        gParameters['conv'] = [flat[i:i+3] for i in range(0, len(flat), 3)]
        print('conv input', gParameters['conv'])

    # build file names for logs and outputs
    ext = benchmark.extension_from_parameters(gParameters, '.keras')
    logfile = gParameters['logfile'] if gParameters['logfile'] else gParameters['output_dir'] + ext + '.log'

    # set up logging (file + console)
    fh = logging.FileHandler(logfile)
    fh.setFormatter(logging.Formatter("[%(asctime)s %(process)d] %(message)s"))
    fh.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(''))
    sh.setLevel(logging.DEBUG if gParameters['verbose'] else logging.INFO)

    benchmark.logger.setLevel(logging.DEBUG)
    benchmark.logger.addHandler(fh)
    benchmark.logger.addHandler(sh)

    benchmark.logger.info(f'params: {gParameters}')

    # get keras defaults and seed
    kerasDefaults = candle.keras_default_config()
    seed = gParameters['rng_seed']

    # load dataset
    loader = benchmark.DataLoader(
        seed=seed,
        dtype=gParameters['data_type'],
        val_split=gParameters['val_split'],
        test_cell_split=gParameters['test_cell_split'],
        cell_features=gParameters['cell_features'],
        drug_features=gParameters['drug_features'],
        feature_subsample=gParameters['feature_subsample'],
        scaling=gParameters['scaling'],
        scramble=gParameters['scramble'],
        min_logconc=gParameters['min_logconc'],
        max_logconc=gParameters['max_logconc'],
        subsample=gParameters['subsample'],
        category_cutoffs=gParameters['category_cutoffs']
    )

    # quick check of input
    print(f"total input dim: {loader.input_dim}")
    print(f"input shapes: {loader.input_shapes}")

    # initialize weights
    initializer_weights = candle.build_initializer(
        gParameters['initialization'], kerasDefaults, seed
    )
    initializer_bias = candle.build_initializer('constant', kerasDefaults, 0.)

    # build model
    model = Sequential()
    gen_shape = None
    out_dim = 1

    # dense model
    if 'dense' in gParameters:
        for layer in gParameters['dense']:
            if layer:
                model.add(Dense(
                    layer,
                    input_dim=loader.input_dim,
                    kernel_initializer=initializer_weights,
                    bias_initializer=initializer_bias
                ))

                # optional batch norm
                if gParameters['batch_normalization']:
                    model.add(BatchNormalization())

                model.add(Activation(gParameters['activation']))

                # optional dropout
                if gParameters['dropout']:
                    model.add(Dropout(gParameters['dropout']))

    # conv model
    else:
        gen_shape = 'add_1d'
        lc_flag = 'locally_connected' in gParameters

        for i, layer_params in enumerate(gParameters['conv']):
            if i == 0:
                add_conv_layer(model, layer_params, input_dim=loader.input_dim, locally_connected=lc_flag)
            else:
                add_conv_layer(model, layer_params, locally_connected=lc_flag)

            if gParameters['batch_normalization']:
                model.add(BatchNormalization())

            model.add(Activation(gParameters['activation']))

            # optional pooling
            if gParameters['pool']:
                model.add(MaxPooling1D(pool_size=gParameters['pool']))

        model.add(Flatten())

    # output layer
    model.add(Dense(out_dim))

    # optimizer
    optimizer = candle.build_optimizer(
        gParameters['optimizer'],
        gParameters['learning_rate'],
        kerasDefaults
    )

    # compile model
    model.compile(loss=gParameters['loss'], optimizer=optimizer)
    model.summary()

    benchmark.logger.debug(f'model: {model.to_json()}')

    # data generators
    train_gen = benchmark.DataGenerator(
        loader,
        batch_size=gParameters['batch_size'],
        shape=gen_shape,
        name='train_gen',
        cell_noise_sigma=gParameters['cell_noise_sigma']
    ).flow()

    val_gen = benchmark.DataGenerator(
        loader,
        partition='val',
        batch_size=gParameters['batch_size'],
        shape=gen_shape,
        name='val_gen'
    ).flow()

    val_gen2 = benchmark.DataGenerator(
        loader,
        partition='val',
        batch_size=gParameters['batch_size'],
        shape=gen_shape,
        name='val_gen2'
    ).flow()

    test_gen = benchmark.DataGenerator(
        loader,
        partition='test',
        batch_size=gParameters['batch_size'],
        shape=gen_shape,
        name='test_gen'
    ).flow()

    # steps per epoch
    train_steps = int(loader.n_train / gParameters['batch_size'])
    val_steps = int(loader.n_val / gParameters['batch_size'])
    test_steps = int(loader.n_test / gParameters['batch_size'])

    # override if provided
    train_steps = gParameters.get('train_steps', train_steps)
    val_steps = gParameters.get('val_steps', val_steps)
    test_steps = gParameters.get('test_steps', test_steps)

    # callbacks
    checkpointer = ModelCheckpoint(
        filepath=gParameters['output_dir'] + '.model' + ext + '.h5',
        save_best_only=True
    )

    progbar = MyProgbarLogger(train_steps * gParameters['batch_size'])

    loss_history = MyLossHistory(
        progbar=progbar,
        val_gen=val_gen2,
        test_gen=test_gen,
        val_steps=val_steps,
        test_steps=test_steps,
        metric=gParameters['loss'],
        category_cutoffs=gParameters['category_cutoffs'],
        ext=ext,
        pre=gParameters['output_dir']
    )

    # set seed for reproducibility
    np.random.seed(seed)

    candleRemoteMonitor = candle.CandleRemoteMonitor(params=gParameters)

    # train model
    history = model.fit_generator(
        train_gen,
        train_steps,
        epochs=gParameters['epochs'],
        validation_data=val_gen,
        validation_steps=val_steps,
        verbose=0,
        callbacks=[checkpointer, loss_history, progbar, candleRemoteMonitor],
    )

    # remove log handlers
    benchmark.logger.removeHandler(fh)
    benchmark.logger.removeHandler(sh)

    print("\nrunning simple xai step...")

    # basic xai step (grab small batches)
    try:
        train_batch = next(train_gen)
        val_batch = next(val_gen)

        # build feature names
        feature_names = []
        for group, shape in loader.input_shapes.items():
            n = int(np.prod(shape))
            feature_names += [f"{group}_{i}" for i in range(n)]

        # select small subsets
        X_bg = np.array(train_batch[0][:100], dtype=np.float32)
        X_explain = np.array(val_batch[0][:70], dtype=np.float32)

        import xai_utils
        xai_utils.run_xai(model, X_bg, X_explain, feature_names=feature_names)

    except Exception as e:
        print(f"xai step failed: {e}")

    return history