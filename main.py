#!/usr/bin/env python2

from misc.util import Struct
import models
import trainers
import worlds

import argparse
import logging
import numpy as np
import os
import random
import sys
import tensorflow as tf
import traceback
import yaml

def main():
    parser = argparse.ArgumentParser('Runs single experiment for a single seed.')
    parser.add_argument('config', nargs='?', type=str, default='config.yaml')
    args = parser.parse_args()
    config = configure(file=args.config)
    world = worlds.load(config)
    model = models.load(config)
    trainer = trainers.load(config)
    tf.compat.v1.disable_eager_execution()
    tf.get_logger().setLevel('INFO') # DEBUG/INFO/....
    trainer.train(model, world)

def configure(file):
    # load config
    with open(file) as config_f:
        config = Struct(**yaml.load(config_f))

    random.seed(config.seed)
    tf.random.set_seed(config.seed)
    tf.compat.v1.set_random_seed(config.seed)
    np.random.seed(config.seed)

    # set up experiment
    config.experiment_dir = os.path.join("experiments/%s" % config.name)
    if not os.path.exists(config.experiment_dir):
        os.mkdir(config.experiment_dir)

    # set up logging
    log_name = os.path.join(config.experiment_dir, "run.log")
    _ = open(log_name, 'w')
    logging.basicConfig(filename=log_name, level=logging.DEBUG,
            format='%(asctime)s %(levelname)-8s %(message)s')
    def handler(type, value, tb):
        logging.exception("Uncaught exception: %s", str(value))
        logging.exception("\n".join(traceback.format_exception(type, value, tb)))
    sys.excepthook = handler

    logging.info("BEGIN")
    logging.info(str(config))

    return config

if __name__ == "__main__":
    main()
