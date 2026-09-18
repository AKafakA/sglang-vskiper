#!/usr/bin/env python3
"""Run the count-GEMM memory contract under an existing compute-sanitizer."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from validate_reference import stop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tree', type=Path, required=True)
    parser.add_argument('--sanitizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise SystemExit('run inside the assigned GPU allocation')
    tree = args.tree.resolve()
    tests = tree / ('vskipper/tests' if (tree / 'vskipper/src').is_dir() else 'test/vp')
    probe_file = tests / 'test_count_gemm_guard.py'
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'job_id': os.environ['SLURM_JOB_ID'],
              'tree': str(tree), 'sanitizer': str(args.sanitizer.resolve()),
              'probe_sha256': hashlib.sha256(probe_file.read_bytes()).hexdigest(), 'checks': []}
    try:
        spec = importlib.util.spec_from_file_location('_count_gemm_memory_probe', probe_file)
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        env = dict(os.environ, PYTORCH_NO_CUDA_MEMORY_CACHING='1', PYTHONDONTWRITEBYTECODE='1')
        paths = [str(tree / 'python')]
        if (tree / 'vskipper/src').is_dir():
            paths.insert(0, str(tree / 'vskipper/src'))
        env['PYTHONPATH'] = os.pathsep.join(paths)
        version = subprocess.run([str(args.sanitizer), '--version'],
                                 capture_output=True, text=True, timeout=30, check=True)
        report['sanitizer_version'] = version.stdout + version.stderr

        def run(name, code, values, negative=False):
            command = [str(args.sanitizer), '--tool', 'memcheck', '--error-exitcode', '99',
                       '--target-processes', 'all', sys.executable, '-c', code, *map(str, values)]
            process = subprocess.Popen(command, env=env, cwd=tree, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                stdout, stderr = process.communicate(timeout=300)
            except BaseException:
                stop(process)  # terminate only this check's owned process group
                raise
            output = stdout + stderr
            (args.output / (name + '.log')).write_text(output)
            if negative:
                passed = (process.returncode != 0 and 'GUARD inside ok' in output
                          and 'Invalid __global__ read' in output and '_peek' in output)
            else:
                passed = (process.returncode == 0 and 'LAUNCH rows=' in output
                          and 'ERROR SUMMARY: 0 errors' in output)
            report['checks'].append({'name': name, 'pass': passed, 'exit_code': process.returncode,
                                     'code_sha256': hashlib.sha256(code.encode()).hexdigest(),
                                     'arguments': list(values)})
            if not passed:
                raise RuntimeError(f'memory check failed: {name}; inspect its log')

        # A planted invalid read proves the checker is active. Disabling the
        # caching allocator gives the checker actual CUDA allocation boundaries.
        run('planted-invalid-read', probe._GUARD, [], negative=True)
        for name, shape in (
            ('qwen-projgd-205', (205, 2560, 1216, 32, 128, 64, 4, 3)),
            ('qwen-projgd-8192', (8192, 2560, 1216, 64, 128, 32, 4, 3)),
            ('qwen-projup-205', (205, 608, 2560, 32, 256, 64, 8, 4)),
            ('llama-projgd-205', (205, 4096, 1792, 32, 128, 64, 4, 3)),
        ):
            run(name, probe._LAUNCH, shape)
        report['status'] = 'pass'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        (args.output / 'status.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
