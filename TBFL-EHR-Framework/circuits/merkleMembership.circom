pragma circom 2.0.0;

include "circomlib/circuits/poseidon.circom";
include "circomlib/circuits/mux1.circom";

// Proves that Poseidon(identitySecret) is one of the leaves of a Merkle tree of
// depth `depth` whose root is the public input `root`, WITHOUT revealing which
// leaf (i.e. which credentialed hospital) it is.
//
// This is the on-chain-verifiable "selective disclosure" primitive described in
// Sec. 4.1 of the paper: an authorized participant proves membership in the
// Verifiable Credential registry without revealing their institutional identity.
//
// `nullifierHash` binds the (secret) identity to a specific round via the public
// `externalNullifier`, so the same credential cannot submit twice in the same
// round -- without that binding, one identitySecret could be replayed anonymously
// forever.
template MerkleMembership(depth) {
    signal input identitySecret;       // private
    signal input pathElements[depth];  // private: sibling hashes on the path to the root
    signal input pathIndices[depth];   // private: 0 = node is left child, 1 = right child

    signal input root;                 // public: current Merkle root of the VC registry
    signal input externalNullifier;    // public: binds the proof to a specific FL round

    signal output nullifierHash;       // public

    component leafHasher = Poseidon(1);
    leafHasher.inputs[0] <== identitySecret;

    component hashers[depth];
    component muxL[depth];
    component muxR[depth];
    signal levelHashes[depth + 1];
    levelHashes[0] <== leafHasher.out;

    for (var i = 0; i < depth; i++) {
        // pathIndices[i] must be boolean
        pathIndices[i] * (1 - pathIndices[i]) === 0;

        muxL[i] = Mux1();
        muxL[i].c[0] <== levelHashes[i];
        muxL[i].c[1] <== pathElements[i];
        muxL[i].s <== pathIndices[i];

        muxR[i] = Mux1();
        muxR[i].c[0] <== pathElements[i];
        muxR[i].c[1] <== levelHashes[i];
        muxR[i].s <== pathIndices[i];

        hashers[i] = Poseidon(2);
        hashers[i].inputs[0] <== muxL[i].out;
        hashers[i].inputs[1] <== muxR[i].out;

        levelHashes[i + 1] <== hashers[i].out;
    }

    root === levelHashes[depth];

    component nullifierHasher = Poseidon(2);
    nullifierHasher.inputs[0] <== identitySecret;
    nullifierHasher.inputs[1] <== externalNullifier;
    nullifierHash <== nullifierHasher.out;
}

// depth=8 supports up to 256 credentialed institutions
component main {public [root, externalNullifier]} = MerkleMembership(8);
