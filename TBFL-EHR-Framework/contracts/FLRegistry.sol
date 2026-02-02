// contracts/FLRegistry.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title Trustworthy Federated Learning Registry (TBFL)
 * @dev Implements the 'Coordination Layer' of the proposed architecture.
 * This contract acts as a gatekeeper, ensuring that only authenticated participants 
 * (holding a simulated Verifiable Credential) can submit model updates.
 */
contract FLRegistry {
    
    // Represents the current state of the Global Model
    struct Task {
        uint id;
        string modelHash; // IPFS Content Identifier (CID) of the global model weights
        uint round;       // Current Federated Learning training round
    }

    // The entity responsible for onboarding hospitals (e.g., Ministry of Health)
    address public trustedIssuer;
    
    // Mapping simulating the Verifiable Credential (VC) verification status.
    // In a production SSI environment, this would verify a Zero-Knowledge Proof (ZKP).
    // For this simulation/PoC, we use an allowlist managed by the Issuer.
    mapping(address => bool) public authorizedWorkers; 

    Task public currentTask;

    // Event emitted when a valid update is accepted. 
    // The Python Aggregator listens for this event to trigger aggregation.
    event ModelUpdated(uint round, string newHash, address indexed contributor);
    event WorkerAuthorized(address indexed worker);

    constructor() {
        // The deployer of the contract is set as the Trusted Issuer (Root of Trust)
        trustedIssuer = msg.sender; 
    }

    /**
     * @dev Simulates the issuance/verification of a Verifiable Credential (VC).
     * @param worker The blockchain address (DID) of the hospital/lab to authorize.
     */
    function authorizeWorker(address worker) external {
        require(msg.sender == trustedIssuer, "Error: Only the Trusted Issuer can authorize nodes.");
        authorizedWorkers[worker] = true;
        emit WorkerAuthorized(worker);
    }

    /**
     * @dev Accepts model updates (gradients) from participants.
     * CRITICAL SECURITY CHECK: This function enforces the "Identity-First" policy.
     * If the caller is not in 'authorizedWorkers', the transaction reverts, preventing Sybil attacks.
     * * @param ipfsHash The IPFS CID pointing to the off-chain stored model weights.
     */
    function submitUpdate(string memory ipfsHash) external {
        // 1. Identity Verification (The core defense mechanism)
        require(authorizedWorkers[msg.sender], "Access Denied: Participant does not hold a valid VC.");
        
        // 2. State Update (Simplified for Simulation)
        // In a full production system, this would wait for 'n' updates before advancing the round.
        currentTask.modelHash = ipfsHash; 
        currentTask.round++;
        
        // 3. Emit Event for Off-Chain Aggregator
        emit ModelUpdated(currentTask.round, ipfsHash, msg.sender);
    }
}
