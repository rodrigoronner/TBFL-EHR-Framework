// Reads {leaves: [...]} from stdin, prints {root} to stdout.
// This is what the Trusted Issuer runs whenever a credential is added to or
// removed from the anonymous registry, before calling setCredentialRoot on-chain.
const { buildTree } = require("./merkleTree");

const DEPTH = 8;

async function main() {
  const input = JSON.parse(require("fs").readFileSync(0, "utf8"));
  const { root } = await buildTree(input.leaves, DEPTH);
  console.log(JSON.stringify({ root: root.toString() }));
}

main()
  .then(() => process.exit(0))
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
