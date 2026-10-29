require("@nomicfoundation/hardhat-toolbox");

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: "0.8.24",
  networks: {
    // 200 contas: cada execução paralela de run_all.py usa um bloco próprio (--account_offset)
    hardhat: {
      chainId: 31337,
      accounts: { count: 200, accountsBalance: "1000000000000000000000000" },
    },
  },
};
