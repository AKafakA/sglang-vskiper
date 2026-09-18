#!/usr/bin/env python3
"""Bounded reference-layout correctness validation on one allocated GPU.

No performance ranking, package installation or source repair is performed.
The JSON contract supplies exact staged trees, assets and archive hashes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def http(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=900 if data else 5) as response:
        return json.load(response)


def run(command, log, env, cwd, timeout=3600):
    print('RUN', ' '.join(map(str, command)), flush=True)
    with log.open('w') as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   env=env, cwd=cwd, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            stop(process)
            raise
    if code:
        raise RuntimeError(f'command failed ({code}): {log}')


def stop(process):
    # Signal only the process group created by this validation harness.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def source_env(tree):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(('SGLANG_FD', 'SGLANG_VP')) or key == 'SGLANG_MOE_CONFIG_DIR':
            env.pop(key)
    src = tree / 'vskipper/src'
    paths = ([str(src)] if src.is_dir() else []) + [str(tree / 'python')]
    paths += ([str(src / 'vskipper/experiments'), str(src / 'vskipper/analysis')]
              if src.is_dir() else [str(tree / 'test/vp')])
    env['PYTHONPATH'] = os.pathsep.join(paths)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['SGLANG_IS_FLASHINFER_AVAILABLE'] = 'false'
    return env


def native_rows(path):
    with Path(path).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()][:32]
    if len(rows) != 32 or len({r['request_id'] for r in rows}) != 32:
        raise ValueError(f'expected 32 distinct frozen request IDs: {path}')
    if not all(isinstance(r['prompt'], list) and r['prompt'] and
               all(type(t) is int and t >= 0 for t in r['prompt']) for r in rows):
        raise ValueError(f'invalid frozen token inputs: {path}')
    return rows


def test_env(tree, contract):
    env = source_env(tree)
    if contract.get('test_tools'):
        directory = Path(contract['test_tools'])
        allowed = {'pytest', '_pytest', 'py.py', 'pluggy', 'iniconfig', 'wheel', 'bin', '__pycache__'}
        for entry in directory.iterdir():
            if entry.name in allowed:
                continue
            if entry.name.endswith('.dist-info') and entry.name.split('-')[0] in allowed:
                continue
            raise ValueError(f'unexpected package in isolated test tools: {entry.name}')
        env['PYTHONPATH'] += os.pathsep + str(directory)
    return env


def probe(url, rows, concurrency, tokenizer, out):
    out.mkdir()
    save(out / 'before.json', http(url + '/server_info'))
    def one(pair):
        index, row = pair
        rid = f"layout-c{concurrency}-{row['request_id']}"
        payload = {'rid': rid, 'input_ids': row['prompt'],
                   'sampling_params': {'temperature': 0.0, 'top_p': 1.0,
                                       'frequency_penalty': 0.0, 'max_new_tokens': 128},
                   'return_logprob': True, 'top_logprobs_num': 5}
        response = http(url + '/generate', payload)
        save(out / f'{index:02d}.response.json', response)
        meta = response['meta_info']
        ids = [entry[1] for entry in meta.get('output_token_logprobs', [])]
        count = meta.get('completion_tokens')
        finish = meta.get('finish_reason')
        if not ids or type(count) is not int or len(ids) != count or not 0 < count <= 128:
            raise ValueError(f'{rid}: incomplete actual-token accounting')
        if not isinstance(finish, dict) or finish.get('type') not in {'stop', 'length'}:
            raise ValueError(f'{rid}: unsuccessful finish reason {finish}')
        if meta.get('id') != rid:
            raise ValueError(f'{rid}: server did not preserve request identity: {meta.get("id")}')
        logs = [entry[0] for entry in meta['output_token_logprobs']]
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in logs):
            raise ValueError(f'{rid}: non-finite token log probabilities')
        return {'request_id': rid, 'input_ids': row['prompt'], 'output_ids': ids,
                'text': response['text'], 'completion_tokens': count,
                'requested_max_new_tokens': 128, 'effective_output_tokens': len(ids),
                'retokenized_output_tokens': len(tokenizer.encode(response['text'], add_special_tokens=False)),
                'finish_reason': finish, 'diagnostic_truncation': finish['type'] == 'length',
                'output_logprobs': logs}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        records = list(pool.map(one, enumerate(rows)))
    save(out / 'records.json', records)
    after = http(url + '/server_info')
    save(out / 'after.json', after)
    for state in after.get('internal_states', []):
        counters = ((state.get('vp_runtime') or {}).get('fd_c3') or {}).get('counters') or {}
        if counters.get('eager_skip_decode_layer_calls', 0) != 0:
            raise ValueError('routed decode escaped the captured execution contract')
        for decision, body in (('dense_overflow_decisions', 'dense_body_passes'),
                               ('dense_overflow_rows', 'dense_body_rows')):
            if counters.get(decision, 0) != counters.get(body, 0):
                raise ValueError(f'coverage accounting mismatch: {decision}/{body}')
    return records


def serve(contract, label, case, results):
    tree = Path(contract[label]['tree'])
    relocated = (tree / 'vskipper/src').is_dir()
    arm_file = tree / ('vskipper/configs/deploy/active_arm' if relocated else 'deploy/active_arm')
    original_arm = arm_file.read_bytes()
    out = results / label / case['name']
    out.mkdir(parents=True)
    env = source_env(tree)
    env['SGLANG_VP_HOST_CONFIG'] = case.get('host_config', contract['host_config'])
    port = contract.get('port', 32179)
    url = f'http://127.0.0.1:{port}'
    with socket.socket() as check:
        if check.connect_ex(('127.0.0.1', port)) == 0:
            raise RuntimeError(f'port {port} is already occupied')
    cmd = [sys.executable, '-m', 'sglang.launch_server', '--model-path', case['model'],
           '--port', str(port), '--tp-size', '1', '--dtype', case['dtype'],
           '--mem-fraction-static', '0.8', '--attention-backend', 'triton',
           '--prefill-attention-backend', 'triton', '--decode-attention-backend', 'triton']
    save(out / 'command.json', {'argv': cmd, 'source_revision': contract[label]['commit'],
                              'pythonpath': env['PYTHONPATH'], 'arm': case['arm']})
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(case['model'], local_files_only=True)
    rows = native_rows(case['prompts'])
    save(out / 'frozen-prompts.json', rows)
    arm_file.write_text(case['arm'] + '\n')
    with (out / 'server.log').open('w') as log:
        process = subprocess.Popen(cmd, cwd=tree, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 1800
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'{label}/{case["name"]}: server exited; see {out / "server.log"}')
                try:
                    http(url + '/server_info')
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f'{label}/{case["name"]}: startup timeout')
                    time.sleep(3)
            probe(url, rows, 1, tokenizer, out / 'serial')
            probe(url, rows, 8, tokenizer, out / 'concurrent')
            if case['require_routing']:
                gates = tree / ('vskipper/src/vskipper/experiments/gates' if relocated else 'test/vp/gates')
                for mode in ('serial', 'concurrent'):
                    run([sys.executable, str(gates / 'verify_skipping_executed.py'),
                         '--before', str(out / mode / 'before.json'),
                         '--after', str(out / mode / 'after.json')],
                        out / f'{mode}-activation.log', env, tree, timeout=120)
        finally:
            stop(process)
            arm_file.write_bytes(original_arm)
    if label == 'candidate':
        left = json.loads((results / 'control' / case['name'] / 'serial/records.json').read_text())
        right = json.loads((out / 'serial/records.json').read_text())
        # Exact IDs, text, actual counts, finish reasons and token log probabilities.
        if left != right:
            save(out / 'parity-failure.json', {'different_rows': [i for i, (a, b) in enumerate(zip(left, right)) if a != b]})
            raise RuntimeError(f'greedy serial parity failed for {case["name"]}')
        tool = tree / 'vskipper/src/vskipper/experiments/qps_deployment.py'
        spec = importlib.util.spec_from_file_location('validation_identity', tool)
        identity = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(identity)
        before = json.loads((results / 'control' / case['name'] / 'serial/after.json').read_text())
        after = json.loads((out / 'serial/after.json').read_text())
        left_id, right_id = identity.stable_server_identity(before), identity.stable_server_identity(after)
        save(out / 'control-stable-identity.json', left_id)
        save(out / 'candidate-stable-identity.json', right_id)
        if left_id != right_id:
            raise RuntimeError(f'served configuration differs for {case["name"]}')
        for old_state, new_state in zip(before['internal_states'], after['internal_states']):
            old_ladder = (old_state.get('vp_runtime') or {}).get('fd_c3', {}).get('ladder', {})
            new_ladder = (new_state.get('vp_runtime') or {}).get('fd_c3', {}).get('ladder', {})
            for key in ('capture_bs', 'decode_capture_bs_max'):
                if old_ladder.get(key) != new_ladder.get(key):
                    raise RuntimeError(f'realized graph ladder differs: {case["name"]}/{key}')
        save(out / 'parity.json', {'rows': 32, 'serial_exact': True, 'concurrent_completion': 32})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('contract', type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise SystemExit('run only inside the assigned Slurm GPU allocation')
    def terminate(signum, frame):
        raise TimeoutError(f'allocation ending: signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    c = json.loads(args.contract.read_text())
    results = Path(c['results'])
    results.mkdir(parents=True, exist_ok=False)
    save(results / 'contract.json', c)
    status = {'status': 'running', 'job_id': os.environ['SLURM_JOB_ID']}
    try:
        status['phase'] = 'dependency preflight'
        save(results / 'status.json', status)
        analysis_env = dict(os.environ)
        if c.get('analysis_environment'):
            analysis_env['PATH'] = str(Path(c['analysis_environment']) / 'bin') + os.pathsep + analysis_env['PATH']
        analysis_python = shutil.which('python3', path=analysis_env['PATH'])
        if not analysis_python:
            raise RuntimeError('analysis Python is missing')
        run([analysis_python, '-c',
             'import sys,numpy,matplotlib; assert sys.version_info[:2] == (3,12); '
             'print(sys.version); print(numpy.__version__,matplotlib.__version__)'],
            results / 'analysis-environment.txt', analysis_env, results, 120)
        for label in ('control', 'candidate'):
            tree = Path(c[label]['tree'])
            run([sys.executable, '-c',
                 'import pytest,pluggy,iniconfig,packaging,pygments,setuptools,wheel; '
                 'print("pytest",pytest.__version__,pytest.__file__); '
                 'print("pluggy",pluggy.__version__,pluggy.__file__)'],
                results / f'{label}-test-dependencies.txt', test_env(tree, c), tree, 120)
        for label in ('control', 'candidate'):
            if sha(c[label]['archive']) != c[label]['archive_sha256']:
                raise ValueError(f'{label}: source archive checksum mismatch')
            tree = Path(c[label]['tree'])
            with tarfile.open(c[label]['archive']) as archive:
                for member in archive:
                    path = tree / member.name
                    if member.isfile():
                        source = archive.extractfile(member)
                        if hashlib.sha256(source.read()).hexdigest() != sha(path):
                            raise ValueError(f'{label}: staged source differs: {member.name}')
                    elif member.issym() and os.readlink(path) != member.linkname:
                        raise ValueError(f'{label}: staged symlink differs: {member.name}')
        for case in c['cases']:
            native_rows(case['prompts'])  # Input dry-run before any server.
        assets = set(c['assets']) | {case['prompts'] for case in c['cases']}
        for case in c['cases']:
            assets.update(str(p) for p in Path(case['model']).glob('*') if p.is_file())
        asset_hashes = {str(Path(p).resolve()): sha(p) for p in sorted(assets)}
        save(results / 'asset-hashes.json', asset_hashes)
        run([sys.executable, '-m', 'pip', 'freeze'], results / 'environment.txt', dict(os.environ), results, 120)
        run(['nvidia-smi', '-q'], results / 'device.txt', dict(os.environ), results, 120)
        for label in ('control', 'candidate'):
            tree = Path(c[label]['tree'])
            env = source_env(tree)
            (results / label).mkdir(exist_ok=True)
            prefix = 'vskipper/tests' if label == 'candidate' else 'test/vp'
            if label == 'candidate':
                if {str(Path(p).resolve()): sha(p) for p in sorted(assets)} != asset_hashes:
                    raise RuntimeError('shared checkpoint or input assets changed between sources')
            status['phase'] = f'{label}: tests'
            save(results / 'status.json', status)
            run([sys.executable, str(tree / prefix / 'test_package_imports.py')],
                results / label / 'imports.log', env, tree)
            tests = [str(tree / prefix / p) for p in c['required_tests']]
            if label == 'candidate':
                tests.append(str(tree / prefix / 'test_source_layout.py'))
            run([sys.executable, '-m', 'pytest', *tests, '-q',
                 '--junitxml=' + str(results / label / 'tests.xml')],
                results / label / 'tests.log', test_env(tree, c), tree, 3600)
            status['phase'] = f'{label}: serving'
            save(results / 'status.json', status)
            # A control failure ends the job before candidate execution.
            for case in c['cases']:
                print('SERVE', label, case['name'], flush=True)
                serve(c, label, case, results)
            if label == 'candidate':
                wheels = results / 'wheel-build'
                wheels.mkdir()
                run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                     '--no-index', '--no-cache-dir', '--wheel-dir', str(wheels), str(tree / 'vskipper')],
                    results / 'wheel-build.log', test_env(tree, c), tree, 300)
                wheel_files = list(wheels.glob('vskipper-*.whl'))
                if len(wheel_files) != 1:
                    raise RuntimeError('expected exactly one vSkipper wheel')
                with zipfile.ZipFile(wheel_files[0]) as bundle:
                    members = set(bundle.namelist())
                    resources = ['vskipper/runtime/device_roofline.json',
                                 'vskipper/kernels/csrc/cuda_conditional_graph.cu',
                                 'vskipper/integration/sglang/model_runner.py']
                    resources += ['vskipper/kernels/binary_cohort_configs/' + p.name
                                  for p in (tree / 'vskipper/src/vskipper/kernels/binary_cohort_configs').glob('*.json')]
                    for resource in resources:
                        if resource not in members or bundle.read(resource) != (tree / 'vskipper/src' / resource).read_bytes():
                            raise RuntimeError(f'wheel resource missing or changed: {resource}')
                save(results / 'wheel-resources.json', {'wheel_sha256': sha(wheel_files[0]),
                                                       'verified_resources': resources})
        for label in ('control', 'candidate'):
            pack = Path(c[label]['pack'])
            run(['bash', str(pack / 'reproduce_results.sh'), str(results / f'{label}-reproduction')],
                results / f'{label}-reproduction.log', analysis_env, results, 1200)
        env = dict(analysis_env, PACK=c['candidate']['pack'])
        run(['bash', str(Path(c['candidate']['tree']) / 'vskipper/scripts/reproduce_analysis.sh'),
             str(results / 'repository-reproduction')],
            results / 'repository-reproduction.log', env, results, 1200)
        status.update(status='pass', phase='complete')
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        save(results / 'status.json', status)


if __name__ == '__main__':
    main()
