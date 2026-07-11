"""
identity.py

Implements the Identity Layer described in Sec. 4.1 of the paper:

  - DIDs: a participant's Decentralized Identifier is `did:ethr:<address>`, the
    standard ERC-1056 default resolution when no key delegation has occurred --
    the DID *is* the Ethereum address, publicly resolvable to a public key.

  - Verifiable Credentials: the Trusted Issuer signs an EIP-712 typed structured
    credential OFF-CHAIN, via the node's `eth_signTypedData_v4` RPC method for its
    own account (`issue_credential`) -- the same pattern the rest of this codebase
    already uses (BlockchainManager never handles raw private keys either). The
    resulting (expiresAt, signature) pair is the VC's cryptographic "proof": anyone
    can submit it to FLRegistryZK.authorizeWorker, which recovers the signer via
    ecrecover and only accepts it if it resolves to the Issuer's DID/address.
"""


def to_did(address: str) -> str:
    """Returns the did:ethr identifier for an Ethereum address."""
    return f"did:ethr:{address}"


def from_did(did: str) -> str:
    """Recovers the Ethereum address from a did:ethr identifier."""
    if not did.startswith("did:ethr:"):
        raise ValueError(f"Not a did:ethr identifier: {did}")
    return did.split("did:ethr:", 1)[1]


def _credential_typed_data(chain_id: int, verifying_contract: str, subject: str, expires_at: int) -> dict:
    """
    Builds the EIP-712 typed-data structure for a "HospitalCredential" VC, matching
    the CREDENTIAL_TYPEHASH / DOMAIN_SEPARATOR computed on-chain in FLRegistryZK.sol.
    """
    return {
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
                {"name": "verifyingContract", "type": "address"},
            ],
            "HospitalCredential": [
                {"name": "subject", "type": "address"},
                {"name": "expiresAt", "type": "uint256"},
            ],
        },
        "primaryType": "HospitalCredential",
        "domain": {
            "name": "TBFL-FLRegistryZK",
            "version": "1",
            "chainId": chain_id,
            "verifyingContract": verifying_contract,
        },
        "message": {
            "subject": subject,
            "expiresAt": expires_at,
        },
    }


def issue_credential(w3, issuer_address: str, chain_id: int, verifying_contract: str,
                      subject_address: str, validity_seconds: int = 30 * 24 * 3600, now: int = None) -> dict:
    """
    Issues a Verifiable Credential: the Trusted Issuer signs an EIP-712 typed
    message (subject, expiresAt) OFF-CHAIN, via `eth_signTypedData_v4` for its own
    unlocked account. This signature is the VC's cryptographic proof of
    authenticity -- exactly analogous to a W3C VC's `proof` block, just EIP-712
    typed data instead of a JSON-LD signature suite.

    Returns a dict {subject, expiresAt, signature} ready to submit to
    FLRegistryZK.authorizeWorker(subject, expiresAt, signature).
    """
    import time
    if now is None:
        now = int(time.time())
    expires_at = now + validity_seconds

    typed_data = _credential_typed_data(chain_id, verifying_contract, subject_address, expires_at)
    signature = w3.manager.request_blocking("eth_signTypedData_v4", [issuer_address, typed_data])

    return {
        "subject": subject_address,
        "subject_did": to_did(subject_address),
        "expiresAt": expires_at,
        "signature": signature,
    }
