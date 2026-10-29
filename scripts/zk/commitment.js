// Reads {identitySecret} from stdin, prints {commitment} to stdout.
// commitment = Poseidon(identitySecret) is the public leaf value that gets
// added to the on-chain Merkle root -- the secret itself never leaves the holder.
const { poseidonHasher } = require("./merkleTree");

async function main() {
  const input = JSON.parse(require("fs").readFileSync(0, "utf8"));
  const { hash1 } = await poseidonHasher();
  const commitment = hash1(input.identitySecret);
  console.log(JSON.stringify({ commitment: commitment.toString() }));
}

main()
  .then(() => process.exit(0))
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
