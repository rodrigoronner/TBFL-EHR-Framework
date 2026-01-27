// contracts/FLRegistry.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FLRegistry {
    struct Task {
        uint id;
        string modelHash; // IPFS CID do modelo global
        uint round;
    }

    address public trustedIssuer;
    mapping(address => bool) public authorizedWorkers; // Simula a posse da VC
    Task public currentTask;

    event ModelUpdated(uint round, string newHash);

    constructor() {
        trustedIssuer = msg.sender; // Você é o Ministério da Saúde na simulação
    }

    // Função que simula a validação da VC off-chain
    function authorizeWorker(address worker) external {
        require(msg.sender == trustedIssuer, "Only Issuer");
        authorizedWorkers[worker] = true;
    }

    // Apenas trabalhadores com VC válida podem enviar updates
    function submitUpdate(string memory ipfsHash) external {
        require(authorizedWorkers[msg.sender], "Access Denied: No Valid VC");
        // Lógica simplificada de agregação:
        // Na prática, emitiríamos um evento para o Agregador Python ler
        currentTask.modelHash = ipfsHash; 
        currentTask.round++;
        emit ModelUpdated(currentTask.round, ipfsHash);
    }
}