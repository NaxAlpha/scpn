from flask import *
import os
import subprocess
from threading import Lock
import shutil
from uuid import uuid4

app = Flask(__name__)
lock = Lock()

TRAINING_PROCESS = None
DATA_PRE_PROCESS = None
ALLOWED_CONFIG = ['batch_size', 'min_sent_length', 'd_word', 'd_trans', 'd_nt', 'd_hid', 'n_epochs', 'lr', 'grad_clip', 'save_freq', 'lr_decay_factor', 'init_trained_model', 'tree_dropout', 'tree_level_dropout', 'short_batch_downsampling_freq', 'short_batch_threshold', 'seed', 'dev_batches']


def default_response(fx):
    def _fx(*args, **kwargs):
        try:
            res = fx(*args, **kwargs)
            rsp = dict(error=0)
            if isinstance(res, str):
                rsp.update(message=res)
            else:
                rsp.update(result=res)
            return jsonify(**rsp)
        except Exception as ex:
            return jsonify(error=hash(ex.__str__()), message=type(ex) + ':' + ex.__str__())

    return _fx


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
    if TRAINING_PROCESS is not None:
        raise Exception('Training already running.')
    cfg = validate_training_config()
    args = ['--%s=%s' % (key, str(val)) for key, val in cfg.items()]
    with lock:
        TRAINING_PROCESS = subprocess.Popen(['python', 'train_scpn.py'] + args, cwd=os.getcwd(), stdout=open('temp/logs.txt', 'w'))
    return 'Training started successfully!'


@app.route('/train/stop')
@default_response
def stop_training():
    global lock, TRAINING_PROCESS
    if TRAINING_PROCESS is None:
        return 'Training not running.'
    with lock:
        TRAINING_PROCESS.kill()
        TRAINING_PROCESS = None
    return 'Training stopped!'


@app.route('/train/logs')
@default_response
def training_logs():
    with open('temp/logs.txt') as f:
        return f.read()


@app.route('/snapshot')
@default_response
def get_snapshot():
    fn = str(uuid4())
    shutil.copy('models/scpn.pt', 'temp/%s'%fn)
    return dict(snapshot_id=fn)


@app.route('/infer/<model>')
@default_response
def paraphrase(model):
    pass


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, threaded=True)

