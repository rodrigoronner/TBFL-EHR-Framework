"""
zk_bridge.py

Python <-> Node.js bridge for the Zero-Knowledge selective-disclosure layer
(circuits/merkleMembership.circom). There is no mature pure-Python Groth16
prover, so proof generation is delegated to snarkjs (JS) via subprocess; this
module hides that detail behind a plain Python API used by the rest of the
codebase (main_tbfl_simulation.py, sybil_attack_experiment.py, etc.).
"""
import json
import secrets
import subprocess
from pathlib import Path

_ZK_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "zk"


def _run_node(script_name: str, payload: dict) -> dict:
    result = subprocess.run(
        ["node", str(_ZK_SCRIPTS_DIR / script_name)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed:\n{result.stderr}")
    return json.loads(result.stdout)


def new_identity_secret() -> int:
    """Generates a fresh random identity secret for a credential holder."""
    return secrets.randbelow(2 ** 200)


def commitment_of(identity_secret: int) -> str:
    """Poseidon(identitySecret): the public leaf committed to the on-chain Merkle root."""
    return _run_node("commitment.js", {"identitySecret": str(identity_secret)})["commitment"]


def compute_root(leaf_commitments: list) -> str:
    """Computes the Merkle root the Trusted Issuer publishes via setCredentialRoot."""
    return _run_node("root.js", {"leaves": [str(c) for c in leaf_commitments]})["root"]


def generate_membership_proof(identity_secret: int, leaf_commitments: list, leaf_index: int, round_number: int) -> dict:
    """
    Generates a Groth16 proof that Poseidon(identity_secret) is one of
    `leaf_commitments` at position `leaf_index`, with the nullifier bound to
    `round_number` (so the proof can't be replayed in a later round).

    Returns {proofA, proofB, proofC, nullifierHash, root, externalNullifier},
    ready to submit to:
        FLRegistryZK.submitUpdateZK(proofA, proofB, proofC, nullifierHash, ipfsHash)
    """
    return _run_node("prove.js", {
        "identitySecret": str(identity_secret),
        "leaves": [str(c) for c in leaf_commitments],
        "leafIndex": leaf_index,
        "externalNullifier": str(round_number),
    })
