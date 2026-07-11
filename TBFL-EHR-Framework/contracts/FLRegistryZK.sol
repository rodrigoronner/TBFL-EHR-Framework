// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IGroth16Verifier {
    function verifyProof(
        uint[2] calldata _pA,
        uint[2][2] calldata _pB,
        uint[2] calldata _pC,
        uint[3] calldata _pubSignals
    ) external view returns (bool);
}

/**
 * @title FLRegistryZK
 * @dev Implements the full Identity + Coordination Layer described in Sec. 4.1 of the paper:
 *
 *   1. DIDs: each participant's identity IS its Ethereum address, resolvable off-chain as
 *      `did:ethr:<address>` (the ERC-1056 default resolution when no key delegation has
 *      occurred) -- see src/identity.py.
 *
 *   2. Verifiable Credentials: the Trusted Issuer signs an EIP-712 typed credential
 *      (subject, expiresAt) OFF-CHAIN with its private key. `authorizeWorker` recovers the
 *      signer on-chain via ecrecover; only a genuine Issuer signature is accepted. Credentials
 *      carry a real expiration and can be revoked -- neither existed in the plain-allowlist
 *      version of this contract.
 *
 *   3. Zero-Knowledge selective disclosure: `submitUpdateZK` lets a credentialed participant
 *      prove Merkle-tree membership in the VC registry (via a Groth16 proof over
 *      circuits/merkleMembership.circom) WITHOUT revealing which institution it is. A
 *      round-bound nullifier stops the same anonymous credential from submitting twice.
 *
 * Scope note: "anonymous" here means the ZK proof does not reveal WHICH credentialed
 * institution is submitting. The Ethereum transaction sender address is still a public field
 * of the transaction itself unless routed through a relayer/meta-transaction, which is out of
 * scope for this PoC.
 */
contract FLRegistryZK {
    struct Task {
        uint id;
        string modelHash;
        uint round;
    }

    address public trustedIssuer;
    IGroth16Verifier public immutable verifier;

    // --- Verifiable Credential registry (named-identity path) ---
    mapping(address => uint256) public credentialExpiry; // 0 => never issued
    mapping(address => bool) public revoked;

    // --- ZK Merkle-membership registry (anonymous path) ---
    bytes32 public credentialRoot;                   // Merkle root of Poseidon(identitySecret) leaves
    mapping(uint256 => bool) public usedNullifiers;  // nullifierHash => used

    Task public currentTask;

    event WorkerAuthorized(address indexed worker, uint256 expiresAt);
    event WorkerRevoked(address indexed worker);
    event CredentialRootUpdated(bytes32 newRoot);
    event ModelUpdated(uint round, string newHash, address indexed contributor);
    event ModelUpdatedAnonymous(uint round, string newHash, uint256 nullifierHash);

    // EIP-712 domain separator + typehash for the "HospitalCredential" Verifiable Credential
    bytes32 private constant CREDENTIAL_TYPEHASH =
        keccak256("HospitalCredential(address subject,uint256 expiresAt)");
    bytes32 private immutable DOMAIN_SEPARATOR;

    constructor(address verifierAddress) {
        trustedIssuer = msg.sender;
        verifier = IGroth16Verifier(verifierAddress);
        DOMAIN_SEPARATOR = keccak256(
            abi.encode(
                keccak256("EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"),
                keccak256("TBFL-FLRegistryZK"),
                keccak256("1"),
                block.chainid,
                address(this)
            )
        );
    }

    // ---------------- Verifiable Credentials (DID + signed VC) ----------------

    /**
     * @dev Authorizes `worker` using a Verifiable Credential: an EIP-712 signature produced
     * OFF-CHAIN by the Trusted Issuer's private key over (subject, expiresAt). Anyone may
     * relay this transaction (e.g. the hospital itself, after receiving the signed VC from
     * the Issuer's wallet) -- what is checked is that the signature recovers to
     * `trustedIssuer`, exactly as a W3C Verifiable Credential's proof is verified.
     */
    function authorizeWorker(address worker, uint256 expiresAt, bytes calldata signature) external {
        bytes32 structHash = keccak256(abi.encode(CREDENTIAL_TYPEHASH, worker, expiresAt));
        bytes32 digest = keccak256(abi.encodePacked("\x19\x01", DOMAIN_SEPARATOR, structHash));
        address signer = _recoverSigner(digest, signature);

        require(signer == trustedIssuer, "Invalid VC: signature is not from the Trusted Issuer");
        require(expiresAt > block.timestamp, "Invalid VC: credential is already expired");

        credentialExpiry[worker] = expiresAt;
        revoked[worker] = false;
        emit WorkerAuthorized(worker, expiresAt);
    }

    /// @dev Only the Trusted Issuer can revoke a previously issued credential.
    function revokeCredential(address worker) external {
        require(msg.sender == trustedIssuer, "Only the Trusted Issuer can revoke credentials");
        revoked[worker] = true;
        emit WorkerRevoked(worker);
    }

    function isAuthorized(address worker) public view returns (bool) {
        return credentialExpiry[worker] > block.timestamp && !revoked[worker];
    }

    /// @dev Named-identity submission path (equivalent to the original FLRegistry.submitUpdate,
    /// now gated by a real signature-verified, expiring, revocable credential).
    function submitUpdate(string calldata ipfsHash) external {
        require(isAuthorized(msg.sender), "Access Denied: no valid, non-expired, non-revoked VC");
        currentTask.modelHash = ipfsHash;
        currentTask.round++;
        emit ModelUpdated(currentTask.round, ipfsHash, msg.sender);
    }

    // ---------------- Zero-Knowledge selective disclosure ----------------

    /// @dev The Trusted Issuer publishes the current Merkle root of every credentialed
    /// institution's Poseidon(identitySecret) commitment. Rebuilt off-chain (see
    /// scripts/zk/) whenever a credential is added to or removed from the anonymous registry.
    function setCredentialRoot(bytes32 newRoot) external {
        require(msg.sender == trustedIssuer, "Only the Trusted Issuer can update the registry root");
        credentialRoot = newRoot;
        emit CredentialRootUpdated(newRoot);
    }

    /**
     * @dev Submits a model update by proving Merkle-tree membership in the credential
     * registry via a Groth16 proof, WITHOUT revealing which institution is submitting.
     * The circuit's public signals are, in order, [nullifierHash, root, externalNullifier];
     * `externalNullifier` must equal `currentTask.round` so a proof cannot be replayed in a
     * later round, and `nullifierHash` is recorded to block a second anonymous submission
     * from the same credential within the same round.
     */
    function submitUpdateZK(
        uint[2] calldata proofA,
        uint[2][2] calldata proofB,
        uint[2] calldata proofC,
        uint256 nullifierHash,
        string calldata ipfsHash
    ) external {
        require(!usedNullifiers[nullifierHash], "Access Denied: credential already used this round");

        uint256[3] memory pubSignals = [nullifierHash, uint256(credentialRoot), currentTask.round];
        require(
            verifier.verifyProof(proofA, proofB, proofC, pubSignals),
            "Access Denied: invalid Zero-Knowledge membership proof"
        );

        usedNullifiers[nullifierHash] = true;
        currentTask.modelHash = ipfsHash;
        currentTask.round++;
        emit ModelUpdatedAnonymous(currentTask.round, ipfsHash, nullifierHash);
    }

    function _recoverSigner(bytes32 digest, bytes calldata signature) private pure returns (address) {
        require(signature.length == 65, "Invalid signature length");
        bytes32 r;
        bytes32 s;
        uint8 v;
        assembly {
            r := calldataload(signature.offset)
            s := calldataload(add(signature.offset, 32))
            v := byte(0, calldataload(add(signature.offset, 64)))
        }
        return ecrecover(digest, v, r, s);
    }
}
