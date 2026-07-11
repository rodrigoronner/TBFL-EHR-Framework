const hre = require("hardhat");
const fs = require("fs");

async function main() {
  /*
   * Deploys the full DID/VC/ZK identity stack:
   *   1. Groth16Verifier -- the snarkjs-generated verifier for circuits/merkleMembership.circom.
   *   2. FLRegistryZK -- the Coordination Layer contract, wired to the verifier address,
   *      exposing both the named-identity VC path (submitUpdate) and the anonymous
   *      ZK-membership path (submitUpdateZK).
   */
  console.log("🚀 Deploying Groth16Verifier (Merkle-membership circuit)...");
  const Verifier = await hre.ethers.getContractFactory("Groth16Verifier");
  const verifier = await Verifier.deploy();
  await verifier.waitForDeployment();
  const verifierAddress = await verifier.getAddress();
  console.log(`✅ Groth16Verifier deployed to: ${verifierAddress}`);

  console.log("🚀 Deploying FLRegistryZK...");
  const FLRegistryZK = await hre.ethers.getContractFactory("FLRegistryZK");
  const registry = await FLRegistryZK.deploy(verifierAddress);
  await registry.waitForDeployment();
  const registryAddress = await registry.getAddress();
  console.log(`✅ FLRegistryZK deployed to: ${registryAddress}`);

  const addresses = { verifier: verifierAddress, registry: registryAddress };
  fs.writeFileSync("zk_deployment.json", JSON.stringify(addresses, null, 2));
  console.log("\n📝 Addresses written to zk_deployment.json");
  console.log(addresses);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
