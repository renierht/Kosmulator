# No-CC priority queue

User requested insertion after the active Model2 free-DE case, before the remaining original queue. Controller: run_logs/insert_no_cc_before_remainder.py, tmux session no_cc_priority. It SIGSTOPs only the original study wrapper, not the sampler or its workers. The child completes sampling and postprocessing normally. After a successful exit, six no-CC cases run sequentially; then SIGCONT resumes the original wrapper to launch Model2 warm and Model3 cases.

No-CC order: uncoupled fixed EOS (LCDM), uncoupled free DE, uncoupled free DM; Model4 fixed EOS, free DE, free DM. Data: Pantheon+SH0ES + DESI DR2. Unrestricted regime, existing priors and model-derived CLASS rd, 120 walkers, 8 workers, burn3000, cap100000, DE/snooker and existing convergence policy. Initial scatter0.002. Plans: validation/model4_no_cc_plans.jsonlog. Outputs start Model4_EOS_noCC_v1.

If the current child fails, no-CC is not started and the original parent is released to record its failure. If no-CC fails, the remaining original queue stays held for inspection. State/PID identities: run_logs/no_cc_queue_insertion.json. Logs: run_logs/no_cc_queue_insertion.log and run_logs/Model4_EOS_noCC_v1.log. Do not launch duplicate jobs or manually resume the original wrapper while no-CC is running. Rebooting ends this in-memory queue arrangement; inspect checkpoints before restarting.
