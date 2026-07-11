const circomlibjs = require("circomlibjs");

/**
 * Poseidon hasher matching the one used inside circuits/merkleMembership.circom
 * (circomlibjs implements the same Poseidon parameters as circomlib's .circom
 * template, which is what guarantees the JS-side tree matches the circuit).
 */
async function poseidonHasher() {
  const poseidon = await circomlibjs.buildPoseidon();
  const F = poseidon.F;
  return {
    hash1: (a) => F.toObject(poseidon([BigInt(a)])),
    hash2: (a, b) => F.toObject(poseidon([BigInt(a), BigInt(b)])),
  };
}

/**
 * Builds a Merkle tree of the given depth over `leaves` (Poseidon(identitySecret)
 * commitments), padding with zero-leaves up to 2^depth.
 */
async function buildTree(leaves, depth) {
  const { hash2 } = await poseidonHasher();
  const size = 1 << depth;
  if (leaves.length > size) {
    throw new Error(`Too many leaves (${leaves.length}) for tree depth ${depth} (max ${size})`);
  }

  let layer = leaves.map((l) => BigInt(l));
  while (layer.length < size) layer.push(0n);

  const layers = [layer];
  for (let d = 0; d < depth; d++) {
    const next = [];
    for (let i = 0; i < layer.length; i += 2) {
      next.push(hash2(layer[i], layer[i + 1]));
    }
    layers.push(next);
    layer = next;
  }
  return { root: layer[0], layers };
}

/**
 * Returns the sibling path (pathElements/pathIndices) for `leafIndex`, in the
 * exact encoding the circuit expects: pathIndices[i] = 0 means the node at
 * level i is a LEFT child (hash(current, sibling)); 1 means it's a RIGHT child
 * (hash(sibling, current)).
 */
function getMerkleProof(layers, leafIndex, depth) {
  const pathElements = [];
  const pathIndices = [];
  let idx = leafIndex;
  for (let d = 0; d < depth; d++) {
    const layer = layers[d];
    const isRightNode = idx % 2;
    const siblingIdx = isRightNode ? idx - 1 : idx + 1;
    pathElements.push(layer[siblingIdx]);
    pathIndices.push(isRightNode);
    idx = Math.floor(idx / 2);
  }
  return { pathElements, pathIndices };
}

module.exports = { poseidonHasher, buildTree, getMerkleProof };
