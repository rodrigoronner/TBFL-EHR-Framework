#!/usr/bin/env bash
# Regenerates the entire ZK toolchain output from source: compiles the circuit,
# runs a Groth16 trusted setup, and exports the verification key + Solidity
# verifier. None of this is committed to the repo (see .gitignore) because it's
# fully reproducible from circuits/merkleMembership.circom.
#
# NOTE ON TRUSTED SETUP: this generates a *single-contributor* Powers of Tau,
# which is appropriate for local development/research PoC use (this is exactly
# what the paper's experiments run against) but is NOT a real multi-party
# ceremony. Do not reuse this zkey for a production deployment where the toxic
# waste must not be knowable by a single party -- for that, use a public
# ceremony's ptau file (e.g. the Hermez/Polygon or PSE ceremonies) as the input
# to `groth16 setup` instead of generating a fresh one here.
set -euo pipefail
cd "$(dirname "$0")/../.."

CIRCOM_BIN="${CIRCOM_BIN:-circom}"
DEPTH_POWER=13   # circuit has ~5100 constraints; 2^13 = 8192 is comfortably enough

mkdir -p circuits/build zk/ptau zk/keys

echo "== Compiling circuit =="
"$CIRCOM_BIN" circuits/merkleMembership.circom --r1cs --wasm --sym -l node_modules -o circuits/build

echo "== Powers of Tau (single-contributor, PoC-only -- see note above) =="
npx snarkjs powersoftau new bn128 "$DEPTH_POWER" zk/ptau/pot_0000.ptau -v
echo "tbfl-zk-setup-$(date +%s)-$RANDOM" | npx snarkjs powersoftau contribute zk/ptau/pot_0000.ptau zk/ptau/pot_0001.ptau --name="TBFL setup" -v
npx snarkjs powersoftau prepare phase2 zk/ptau/pot_0001.ptau zk/ptau/pot_final.ptau -v

echo "== Groth16 setup =="
npx snarkjs groth16 setup circuits/build/merkleMembership.r1cs zk/ptau/pot_final.ptau zk/keys/merkleMembership_0000.zkey
echo "tbfl-zk-zkey-$(date +%s)-$RANDOM" | npx snarkjs zkey contribute zk/keys/merkleMembership_0000.zkey zk/keys/merkleMembership_final.zkey --name="TBFL zkey" -v
npx snarkjs zkey export verificationkey zk/keys/merkleMembership_final.zkey zk/keys/verification_key.json

echo "== Exporting Solidity verifier (overwrites contracts/MerkleMembershipVerifier.sol) =="
npx snarkjs zkey export solidityverifier zk/keys/merkleMembership_final.zkey contracts/MerkleMembershipVerifier.sol

echo ""
echo "✅ ZK toolchain ready. Since the verifier contract's verification key is now"
echo "   fresh, redeploy: npx hardhat run scripts/deploy_zk.js --network localhost"
