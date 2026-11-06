"""Exercise snapshot reads while an emcee HDFBackend appends in another process."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import multiprocessing as mp
import tempfile
import time
import numpy as np
import emcee
from Kosmulator_main.peek import inspect_chain


def write_chain(path, ready, release):
    rng = np.random.default_rng(91)
    backend = emcee.backends.HDFBackend(path)
    backend.reset(12, 2)
    sampler = emcee.EnsembleSampler(12, 2, lambda x: -.5 * (x @ x), backend=backend)
    state = sampler.run_mcmc(rng.normal(size=(12, 2)), 20)
    ready.set()
    if not release.wait(30):
        raise RuntimeError("Inspector did not start")
    for _ in range(100):
        state = sampler.run_mcmc(state, 1)
        time.sleep(.01)


if __name__ == "__main__":
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="ide-live-peek-") as folder:
        path = str(Path(folder)/"chain.h5")
        ready, release = context.Event(), context.Event()
        writer = context.Process(target=write_chain, args=(path, ready, release))
        writer.start()
        successes, copy_errors = 0, []
        try:
            assert ready.wait(30)
            release.set()
            for _ in range(20):
                try:
                    report = inspect_chain(path, 4)
                    assert report["retained_steps"] == report["saved_steps"] - 5
                    successes += 1
                except OSError as exc:
                    # Non-atomic snapshots can race metadata updates.
                    copy_errors.append(str(exc))
                time.sleep(.02)
            writer.join(30)
            assert writer.exitcode == 0, writer.exitcode
            final = inspect_chain(path, 4)
            assert final["saved_steps"] == 120
            assert successes > 0
            report = dict(successful_snapshots=successes, transient_copy_errors=copy_errors,
                          writer_exitcode=writer.exitcode, final_saved_steps=120)
            print(json.dumps(report, indent=2))
        finally:
            if writer.is_alive():
                writer.terminate()
                writer.join()
