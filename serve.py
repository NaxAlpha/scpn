from flask import *
import os
import subprocess
from threading import Lock

app = Flask(__name__)
lock = Lock()

TRAINING_PROCESS = None
DATA_PRE_PROCESS = None
ALLOWED_CONFIG = ['batch_size', 'min_sent_length', 'd_word', 'd_trans', 'd_nt', 'd_hid', 'n_epochs', 'lr', 'grad_clip', 'save_freq', 'lr_decay_factor', 'init_trained_model', 'tree_dropout', 'tree_level_dropout', 'short_batch_downsampling_freq', 'short_batch_threshold', 'seed', 'dev_batches']


def default_response(fx):
    def _fx(*args, **kwargs):
        try:
            return jsonify(error=0, message=fx(*args, **kwargs))
        except Exception as ex:
            return jsonify(error=hash(ex.__str__()), message=type(ex) + ':' + ex.__str__())

    return fx


def validate_training_config():
    global ALLOWED_CONFIG
    cfg = request.get_json()
    for key in cfg:
        if key not in ALLOWED_CONFIG:
            raise Exception('Key %s is not recognized!' % key)
        if not isinstance(cfg[key], (str, int, float)):
            raise Exception('Key %s has wired type %s' % (key, type(key)))
    return cfg


@app.route('/train/start', methods=['POST'])
@default_response
def start_training():
    global lock, TRAINING_PROCESS
    cfg = validate_training_config()
    args = ['--%s=%s' % (key, str(val)) for key, val in cfg.items()]
    with lock:
        TRAINING_PROCESS = subprocess.Popen(['python', 'train_scpn.py'] + args, cwd=os.getcwd(), stdout=open('temp/logs.txt', 'w'))
    return 'Training started successfully!'


@app.route('/train/stop')
@default_response
def stop_training():
    global lock, TRAINING_PROCESS
    with lock:
        TRAINING_PROCESS.kill()
        TRAINING_PROCESS = None
    return 'Training stopped!'


@app.route('/train/logs')
def training_logs():
    with open('temp/logs.txt') as f:
        return f.read()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, threaded=True)

