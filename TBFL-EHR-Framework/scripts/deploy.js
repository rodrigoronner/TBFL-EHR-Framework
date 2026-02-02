const hre = require("hardhat");

async function main() {
  /*
   * Deployment Script for the FLRegistry Smart Contract.
   * * This script initializes the "Coordination Layer" of the TBFL framework
   * on the target blockchain network (e.g., Hardhat Local Node or Hyperledger Besu).
   * It deploys the immutable registry used for DID verification and Model Hash logging.
   */

  console.log("🚀 Starting deployment of FLRegistry Smart Contract...");

  // 1. Retrieve the Contract Factory
  // The ContractFactory is an abstraction used to deploy new smart contracts,
  // acting as a factory for instances of the FLRegistry class.
  const FLRegistry = await hre.ethers.getContractFactory("FLRegistry");
  
  // 2. Deploy the Contract
  // This sends a signed transaction to the network to create the contract instance.
  // At this stage, the 'constructor' of the Solidity contract is executed.
  const registry = await FLRegistry.deploy();
  
  // 3. Wait for Block Confirmation
  // Ensures the transaction has been mined, validated, and included in a block 
  // by the consensus mechanism.
  await registry.waitForDeployment();

  const address = await registry.getAddress();
  console.log(`✅ FLRegistry successfully deployed to: ${address}`);
  console.log(`\n⚠️  IMPORTANT: Copy this address and update 'CONTRACT_ADDRESS' in 'blockchain_manager.py'.`);
}

// Standard Hardhat error handling pattern to ensure process exit codes are correct
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
