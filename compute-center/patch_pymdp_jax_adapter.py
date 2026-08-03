#!/usr/bin/env python3
"""One-time deterministic migration of the pymdp adapter to the JAX-first API."""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
PATH = HERE / "behavior_finance_intelligence_operations.py"

OLD = '''    A = utils.obj_array(1)
    B = utils.obj_array(1)
    C = utils.obj_array(1)
    D = utils.obj_array(1)
    A[0] = likelihood
    B[0] = np.stack(transitions, axis=2)
    C[0] = preferences
    D[0] = prior
    agent = Agent(A=A, B=B, C=C, D=D, policy_len=1)
    posterior_states = agent.infer_states([observation])
    policy_probabilities, negative_expected_free_energy = agent.infer_policies()
    q_pi = np.asarray(policy_probabilities, dtype=float).reshape(-1)
    chosen_policy = int(np.argmax(q_pi))
    policy = np.asarray(agent.policies[chosen_policy], dtype=int).reshape(-1)
    chosen_action = int(policy[0])
    posterior = np.asarray(posterior_states[0], dtype=float).reshape(-1)
    negative_efe = np.asarray(negative_expected_free_energy, dtype=float).reshape(-1)
'''

NEW = '''    from jax import numpy as jnp

    A = [jnp.asarray(likelihood, dtype=jnp.float32)]
    B = [jnp.asarray(np.stack(transitions, axis=2), dtype=jnp.float32)]
    C = [jnp.asarray(preferences, dtype=jnp.float32)]
    D = [jnp.asarray(prior, dtype=jnp.float32)]
    agent = Agent(A=A, B=B, C=C, D=D, policy_len=1, batch_size=1)
    posterior_states, inference_info = agent.infer_states(
        [jnp.asarray([observation], dtype=jnp.int32)],
        empirical_prior=agent.D,
        return_info=True,
    )
    policy_probabilities, negative_expected_free_energy = agent.infer_policies(posterior_states)
    q_pi = np.asarray(policy_probabilities, dtype=float).reshape(agent.batch_size, -1)[0]
    chosen_policy = int(np.argmax(q_pi))
    policy_array = np.asarray(agent.policies.policy_arr, dtype=int)
    chosen_action = int(policy_array[chosen_policy, 0, 0])
    posterior = np.asarray(posterior_states[0], dtype=float)[0, -1]
    negative_efe = np.asarray(negative_expected_free_energy, dtype=float).reshape(agent.batch_size, -1)[0]
    vfe = np.asarray(inference_info["vfe"], dtype=float).reshape(-1)
'''

text = PATH.read_text(encoding="utf-8")
if OLD in text:
    text = text.replace(OLD, NEW, 1)
elif NEW not in text:
    raise SystemExit("pymdp adapter source block not found")

old_result = '''        "negative_expected_free_energy": negative_efe.tolist(),
        "engine": {"inferactively-pymdp": _package("inferactively-pymdp")},
'''
new_result = '''        "negative_expected_free_energy": negative_efe.tolist(),
        "variational_free_energy": vfe.tolist(),
        "engine": {"inferactively-pymdp": _package("inferactively-pymdp")},
'''
if old_result in text:
    text = text.replace(old_result, new_result, 1)
elif new_result not in text:
    raise SystemExit("pymdp result block not found")

PATH.write_text(text, encoding="utf-8")
print("PYMDP_JAX_ADAPTER_PATCHED")
