const hre = require("hardhat");

async function main() {
  // Pega a "fábrica" do contrato
  const FLRegistry = await hre.ethers.getContractFactory("FLRegistry");
  
  // Faz o deploy
  const registry = await FLRegistry.deploy();
  
  // Aguarda ser minerado
  await registry.waitForDeployment();

  console.log(`FLRegistry deployed to: ${await registry.getAddress()}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});