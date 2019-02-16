from flask import *
import subprocess
from threading import Lock

app = Flask(__name__)
lock = Lock()

TRAINING_PROCESS = None
DATA_PRE_PROCESS = None
ALLOWED_CONFIG = ['batch_size', 'min_sent_length', 'd_word', 'd_trans', 'd_nt', 'd_hid', 'n_epochs', 'lr', 'grad_clip', 'save_freq', 'lr_decay_factor', 'init_trained_model', 'tree_dropout', 'tree_level_dropout', 'short_batch_downsampling_freq', 'short_batch_threshold', 'seed', 'dev_batches']


@app.route('/train/start', methods=['POST'])
def start_training():
    global lock, TRAINING_PROCESS, ALLOWED_CONFIG
    if TRAINING_PROCESS is not None:
        return jsonify(error=1, message='Training already running.')
    cfg = request.get_json()
    for key in cfg:
        if key not in ALLOWED_CONFIG:
            return jsonify(error=2, message='Key %s is not recognized!' % key)
        if not isinstance(cfg[key], (str, int, float)):
            return jsonify(error=3, message='Key %s has wired type %s' % (key, type(key)))
    
    args = ' '.join('--%s=%s' % (key, str(val)) for key, val in cfg.items())
    with lock:
        TRAINING_PROCESS = subprocess.Popen('python train.py %s >'%args)
    return jsonify(error=0, message='Training started successfully!')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, threaded=True)

