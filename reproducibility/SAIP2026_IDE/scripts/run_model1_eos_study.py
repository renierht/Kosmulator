#!/usr/bin/env python3
"""Plan three matched two-coupling Model 1 runs; sampling requires --sample."""
import argparse
import subprocess
import sys
from pathlib import Path

CASES = {
    f'model{model}_{label}': (f'Linear_IDE_{model}', eos)
    for model in (1,)
    for label, eos in (('lambda', 'cold_lambda'), ('free_de', 'cold_free_de'), ('warm_lambda', 'warm_lambda'))
}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', required=True)
    p.add_argument('--case', choices=('all', *CASES), default='all')
    p.add_argument('--w-dm-prior', nargs=2, type=float, required=True, metavar=('LOW','HIGH'),
                   help='Explicit scientific prior; no default is assumed')
    p.add_argument('--delta-dm-prior', nargs=2, type=float, required=True)
    p.add_argument('--delta-de-prior', nargs=2, type=float, required=True)
    p.add_argument('--num-cores', type=int, default=8)
    p.add_argument('--include-cc', action='store_true', help='Add CC consistently to all selected cases')
    p.add_argument('--sample', action='store_true', help='Start sampling; otherwise print plans only')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--burn', type=int, default=3000, help='Discard these initial steps; use the original value when resuming an existing run')
    a = p.parse_args()
    if not 0 <= a.burn < 100000:
        p.error('Require 0 <= burn < 100000')
    driver = Path(__file__).with_name('run_model1_eos_analysis.py')
    cases = CASES if a.case == 'all' else {a.case: CASES[a.case]}
    # Validate every command before starting any sampling. Independent directories;
    # sequential execution prevents six worker pools competing for the same RAM.
    commands = []
    for name, (model, eos) in cases.items():
        command = [sys.executable, '-u', str(driver), '--regime', 'iw',
                   '--only-model', 'ide', '--ide-model', model,
                   '--eos-case', eos, '--run-name', a.run_name+'_'+name,
                   '--num-cores', str(a.num_cores), '--max-steps', '100000',
                   '--burn', str(a.burn), '--init-jitter-frac', '0.002', '--emcee-moves', 'de_snooker']
        command += ['--delta-dm-prior', *map(str,a.delta_dm_prior), '--delta-de-prior', *map(str,a.delta_de_prior)]
        if a.include_cc: command.append('--include-cc')
        if eos == 'warm_lambda': command += ['--w-dm-prior', *map(str,a.w_dm_prior)]
        if a.resume: command.append('--resume')
        subprocess.run(command+['--plan'], check=True)
        commands.append(command)
    if a.sample:
        for command in commands: subprocess.run(command, check=True)

if __name__ == '__main__':
    main()
