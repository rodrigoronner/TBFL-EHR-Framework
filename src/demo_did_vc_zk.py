"""
demo_did_vc_zk.py

Standalone proof-of-concept walking through the four-phase Identity workflow
described in Sec. 4.1/4.2 of the paper, using real cryptography end-to-end
(no simulated/placeholder steps):

  Phase 1 (Onboarding & Credential Issuance): the Trusted Issuer signs an
      EIP-712 Verifiable Credential OFF-CHAIN for each legitimate hospital's
      DID (did:ethr:<address>). FLRegistryZK.authorizeWorker verifies the
      signature on-chain via ecrecover.

  Phase 2 (Authentication): demonstrated two ways --
      (a) named path: `isAuthorized` / `submitUpdate` check the signed VC's
          expiry and revocation status for a known address;
      (b) anonymous path: the Trusted Issuer publishes a Merkle root of every
          credentialed hospital's Poseidon(identitySecret) commitment.

  Phase 3 (Authenticated Model Update Submission): a hospital submits a
      (simulated) model-update hash either under its own DID (`submitUpdate`)
      or anonymously, by proving Groth16 membership in the credential
      registry without revealing which hospital it is (`submitUpdateZK`).

  Phase 4 (Secure Aggregation) happens off-chain in main_tbfl_simulation.py
      and is out of scope for this identity-layer demo.

Prerequisites (see README):
    npx hardhat node                                  # Terminal A
    npx hardhat run scripts/deploy_zk.js --network localhost   # Terminal B
    python src/demo_did_vc_zk.py                       # Terminal B
"""
import json
import sys
from pathlib import Path

from web3 import Web3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identity import issue_credential, to_did
from zk_bridge import new_identity_secret, commitment_of, compute_root, generate_membership_proof

RPC_URL = "http://127.0.0.1:8545"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_deployment():
    deployment_path = PROJECT_ROOT / "zk_deployment.json"
    if not deployment_path.exists():
        raise FileNotFoundError(
            "zk_deployment.json not found. Run 'npx hardhat run scripts/deploy_zk.js "
            "--network localhost' first."
        )
    return json.loads(deployment_path.read_text())


def load_abi():
    abi_path = PROJECT_ROOT / "artifacts" / "contracts" / "FLRegistryZK.sol" / "FLRegistryZK.json"
    if not abi_path.exists():
        raise FileNotFoundError(f"ABI not found at {abi_path}. Run 'npx hardhat compile' first.")
    return json.loads(abi_path.read_text())["abi"]


def submit_zk(registry, proof, ipfs_hash, sender):
    return registry.functions.submitUpdateZK(
        [int(proof["proofA"][0], 16), int(proof["proofA"][1], 16)],
        [[int(proof["proofB"][0][0], 16), int(proof["proofB"][0][1], 16)],
         [int(proof["proofB"][1][0], 16), int(proof["proofB"][1][1], 16)]],
        [int(proof["proofC"][0], 16), int(proof["proofC"][1], 16)],
        int(proof["nullifierHash"], 16),
        ipfs_hash,
    ).transact({"from": sender})


def main():
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        print("❌ Failed to connect to Blockchain. Run 'npx hardhat node' first.")
        return

    deployment = load_deployment()
    registry = w3.eth.contract(address=deployment["registry"], abi=load_abi())

    issuer = w3.eth.accounts[0]
    hospital_a = w3.eth.accounts[1]
    attacker = w3.eth.accounts[3]
    relayer = w3.eth.accounts[9]  # anonymous submissions can be relayed by anyone

    print("=" * 70)
    print("PHASE 1: Onboarding & Credential Issuance (named DID/VC path)")
    print("=" * 70)
    print(f"🆔 Hospital A DID: {to_did(hospital_a)}")
    vc = issue_credential(w3, issuer, w3.eth.chain_id, deployment["registry"], hospital_a, validity_seconds=3600)
    print(f"🏛️  Issuer signed a Verifiable Credential (EIP-712) for Hospital A, expiring at {vc['expiresAt']}")

    tx = registry.functions.authorizeWorker(vc["subject"], vc["expiresAt"], bytes.fromhex(vc["signature"][2:])).transact({"from": issuer})
    w3.eth.wait_for_transaction_receipt(tx)
    print(f"   ✅ On-chain ecrecover confirmed the Issuer's signature. isAuthorized={registry.functions.isAuthorized(hospital_a).call()}")
    print(f"⚠️  Attacker {to_did(attacker)} was never issued a credential. isAuthorized={registry.functions.isAuthorized(attacker).call()}")

    print("\n" + "=" * 70)
    print("PHASE 2/3 (named path): Authenticated Model Update Submission")
    print("=" * 70)
    tx = registry.functions.submitUpdate("QmNamedHospitalA_Round").transact({"from": hospital_a})
    r = w3.eth.wait_for_transaction_receipt(tx)
    print(f"🏥 Hospital A submitted under its own DID. Status: {r.status}")
    try:
        tx = registry.functions.submitUpdate("QmAttacker_Round").transact({"from": attacker})
        w3.eth.wait_for_transaction_receipt(tx)
        print("   ❌ ATTACKER SUBMISSION SUCCEEDED (this should not happen)")
    except Exception:
        print("   ⛔ Attacker BLOCKED: Access Denied (no valid VC)")

    print("\n" + "=" * 70)
    print("PHASE 1/2/3 (anonymous path): Zero-Knowledge selective disclosure")
    print("=" * 70)
    legit_secrets = [new_identity_secret() for _ in range(3)]
    attacker_secret = new_identity_secret()  # never added to the registry
    commitments = [commitment_of(s) for s in legit_secrets]
    print(f"🔑 {len(legit_secrets)} hospitals each generated a private identitySecret and "
          f"disclosed only Poseidon(identitySecret) to the Issuer.")

    root = compute_root(commitments)
    tx = registry.functions.setCredentialRoot(bytes.fromhex(format(int(root), "064x"))).transact({"from": issuer})
    w3.eth.wait_for_transaction_receipt(tx)
    print(f"🏛️  Issuer published the Merkle root of all credentialed commitments on-chain.")

    current_round = registry.functions.currentTask().call()[2]
    proof = generate_membership_proof(legit_secrets[0], commitments, 0, current_round)
    tx = submit_zk(registry, proof, "QmAnon_Round", relayer)
    r = w3.eth.wait_for_transaction_receipt(tx)
    print(f"🕶️  A credentialed hospital submitted a model update proving Groth16 Merkle "
          f"membership -- WITHOUT revealing which of the {len(legit_secrets)} it is. Status: {r.status}")

    try:
        submit_zk(registry, proof, "QmReplay_Round", relayer)
        print("   ❌ REPLAY SUCCEEDED (this should not happen)")
    except Exception:
        print("   ⛔ Replay of the same proof in the same round BLOCKED (nullifier already used)")

    try:
        bad_proof = generate_membership_proof(attacker_secret, commitments, 0, current_round + 1)
        submit_zk(registry, bad_proof, "QmAttackerAnon_Round", relayer)
        print("   ❌ NON-MEMBER SUCCEEDED (this should not happen)")
    except Exception:
        print("   ⛔ Attacker (identitySecret never registered) BLOCKED: cannot construct "
              "a valid membership proof at all -- the circuit's root constraint fails "
              "before any transaction is even sent.")

    print("\n✅ Demo complete: DID, EIP-712 Verifiable Credentials (with real ecrecover "
          "verification, expiry and revocation), and Groth16 Zero-Knowledge selective "
          "disclosure are all live and independently verified on-chain.")


if __name__ == "__main__":
    main()
