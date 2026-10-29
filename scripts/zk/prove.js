// Reads {identitySecret, leaves, leafIndex, externalNullifier} from stdin,
// generates a Groth16 membership proof and prints the calldata as JSON to stdout:
// {proofA, proofB, proofC, nullifierHash, root, externalNullifier}.
const path = require("path");
const fs = require("fs");
const snarkjs = require("snarkjs");
const { buildTree, getMerkleProof } = require("./merkleTree");

const DEPTH = 8;
const WASM_PATH = path.join(__dirname, "../../circuits/build/merkleMembership_js/merkleMembership.wasm");
const ZKEY_PATH = path.join(__dirname, "../../zk/keys/merkleMembership_final.zkey");

async function main() {
  const input = JSON.parse(fs.readFileSync(0, "utf8"));

  const { root, layers } = await buildTree(input.leaves, DEPTH);
  const { pathElements, pathIndices } = getMerkleProof(layers, input.leafIndex, DEPTH);

  const circuitInput = {
    identitySecret: input.identitySecret,
    pathElements: pathElements.map(String),
    pathIndices: pathIndices.map(String),
    root: root.toString(),
    externalNullifier: input.externalNullifier,
  };

  const { proof, publicSignals } = await snarkjs.groth16.fullProve(circuitInput, WASM_PATH, ZKEY_PATH);

  // Reuse snarkjs's own (tested) Solidity calldata formatting instead of
  // hand-rolling the G2 coordinate reordering, which is a well-known footgun.
  const calldata = await snarkjs.groth16.exportSolidityCallData(proof, publicSignals);
  const [pA, pB, pC, pubSignals] = JSON.parse("[" + calldata + "]");

  console.log(JSON.stringify({
    proofA: pA,
    proofB: pB,
    proofC: pC,
    nullifierHash: pubSignals[0],
    root: pubSignals[1],
    externalNullifier: pubSignals[2],
  }));
}

main()
  .then(() => process.exit(0))
  // snarkjs's WASM bn128 curve init leaves open handles that keep the process
  // alive after main() resolves; force exit so subprocess callers don't hang.
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
