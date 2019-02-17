from flask import *
import os
import subprocess
from threading import Lock
import shutil
from uuid import uuid4
from functools import wraps
import json
import csv
from subwordnmt.apply_bpe import BPE, read_vocabulary
import numpy as np
import cPickle
import h5py

app = Flask(__name__)
lock = Lock()

TRAINING_PROCESS = None
DATA_PRE_PROCESS = None
ALLOWED_CONFIG = ['snapshot_id', 'batch_size', 'min_sent_length', 'd_word', 'd_trans', 'd_nt', 'd_hid', 'n_epochs', 'lr', 'grad_clip', 'save_freq', 'lr_decay_factor', 'init_trained_model', 'tree_dropout', 'tree_level_dropout', 'short_batch_downsampling_freq', 'short_batch_threshold', 'seed', 'dev_batches']


def temp_file(fn=None):
    if fn is None:
        fn = str(uuid4())
    return os.path.join('temp', fn)


def default_response(fx):

    @wraps(fx)
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
            return jsonify(error=hash(ex.__str__()), message='%s: %s' % (type(ex).__name__, ex))

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


def process_data(fin):
    curdir = os.getcwd()
    os.chdir('/data/nlp/')
    cmd = 'java -Xmx12g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLP -threads 8 -annotators tokenize,ssplit,pos,parse -ssplit.eolonly -file /data/scpn/{} -outputFormat text -parse.model edu/stanford/nlp/models/srparser/englishSR.ser.gz -outputDirectory /data/scpn/temp/'.format(fin)
    ret = os.system(cmd)
    os.chdir(curdir)
    shutil.move(fin + '.out', 'data/paranmt_dev_parses.txt')
    cmd = 'python read_paranmt_parses.py'
    ret = os.system(cmd)
    fout = str(uuid4())
    with open('data/parsed_paranmt.csv', 'r') as f:
        reader = csv.DictReader(f, delimiter='\t', fieldnames=['tokens', 'parse'])
        return [row for row in reader][1:]


def make_trainable(data, bpe, vocab):
    data_file = temp_file()
    with open(data_file, 'w') as f:
        f.write('\n'.join(data))
    data_parses = process_data(data_file)
    processed = []
    lengths = []
    parses = []
    for item in data_parses:
        k = bpe.segment(item['tokens'].lower()).split()
        k = [vocab[i] for i in k if i in vocab] + [vocab['EOS']]
        lengths.append(len(k))
        parses.append(item['parse'])
        [k.append(0) for _ in range(40-len(k))]
        processed.append(np.array(k, dtype='int16'))
    return np.array(processed, dtype='int16'), \
        np.array(lengths, dtype='|u1'), \
        np.array(parses)


@app.route('/train/start', methods=['POST'])
@default_response
def start_training():
    global lock, TRAINING_PROCESS
    if TRAINING_PROCESS is not None:
        raise Exception('Training already running.')
    cfg = validate_training_config()
    try:
        snapshot = cfg.pop('snapshot')
        shutil.copy(temp_file(snapshot), 'models/scpn.pt')
        cfg['init_trained_model'] = 1
    except KeyError:
        pass
    except FileNotFoundError:
        raise Exception('Snapshot %s not found'%snapshot)
    args = ['--%s=%s' % (key, str(val)) for key, val in cfg.items()]
    with lock:
        TRAINING_PROCESS = subprocess.Popen(['python', 'train_scpn.py'] + args, cwd=os.getcwd(), stdout=open(temp_file('logs.txt'), 'w'))
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
    with open(temp_file('logs.txt')) as f:
        return dict(logs=f.read())


@app.route('/snapshot')
@default_response
def get_snapshot():
    fn = str(uuid4())
    shutil.copy('models/scpn.pt', temp_file(fn))
    return dict(snapshot_id=fn)


@app.route('/train/data', methods=['POST'])
@default_response
def add_training_data():
    content = request.get_json()

    with open('data/vocab.txt') as vocab:
        vocab = read_vocabulary(vocab, 50)
    with open('data/bpe.codes') as codes:
        bpe = BPE(codes, vocab=vocab)
    with open('data/parse_vocab.pkl' , 'rb') as fx:
        pp_vocab, _ = cPickle.load(fx)

    inputs, in_lengths, input_parses = make_trainable(content.keys(), bpe, pp_vocab)
    outputs, out_lengths, output_parses = make_trainable(content.values(), bpe, pp_vocab)
    with h5py.File('data/parsed_data2.h5') as g:
        with h5py.File('data/parsed_data.h5') as f:
            g.create_dataset('inputs', data=np.concatenate((f['inputs'], inputs)))
            g.create_dataset('outputs', data=np.concatenate((f['outputs'], outputs)))
            g.create_dataset('in_lengths', data=np.concatenate((f['in_lengths'], in_lengths)))
            g.create_dataset('out_lengths', data=np.concatenate((f['out_lengths'], out_lengths)))
            g.create_dataset('input_parses', data=np.concatenate((f['input_parses'], input_parses)))
            g.create_dataset('output_parses', data=np.concatenate((f['output_parses'], output_parses)))


@app.route('/infer/<model>', methods=['POST'])
@default_response
def paraphrase(model):
    pass


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

