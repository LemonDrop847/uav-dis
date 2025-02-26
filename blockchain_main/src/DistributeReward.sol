// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.24;

import "./Contributor.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract DistributeReward is Ownable {
    Contributor public contributor;

    mapping(address => uint256) public rewardsClaimed;
    mapping(address => uint256) public lastClaimedReward;
    mapping(address => uint256) public rewardAddress;

    event RewardDistributed(address indexed device, uint256 amount);
    event ConributorSet(address indexed contributor);
    event RewardDeposited(address indexed device, uint256 amount);
    event RewardClaimed(address indexed device, uint256 amount);

    constructor() Ownable(msg.sender) {}

    function setContributor(address _contributor) external onlyOwner {
        contributor = Contributor(_contributor);
        emit ConributorSet(_contributor);
    }

    function depositReward() external payable {
        require(msg.value > 0, "Must deposit a positive value");
        emit RewardDeposited(msg.sender, msg.value);
    }

    function calculateReward() external {
        uint256 totalWeight = contributor.getTotalWeight();
        uint256 balance = address(this).balance;
        require(totalWeight > 0, "No contributions made");

        address[] memory deviceList = contributor.getDeviceList();

        for (uint256 i = 0; i < deviceList.length; i++) {
            address device = deviceList[i];
            uint256 deviceContribution = contributor.getContribution(device);
            require(deviceContribution > 0, "No contribution from device");
            uint256 rewardAmount = (balance * deviceContribution) / totalWeight;
            rewardAddress[device] = rewardAmount;
        }
    }

    function distributeReward(address _device) external {
        uint256 userContribution = contributor.getContribution(_device);
        require(userContribution > 0, "No contribution from device");
        payable(_device).transfer(rewardAddress[_device]);
        emit RewardDistributed(_device, rewardAddress[_device]);
    }

    function getRewardPoolBalance() external view returns (uint256) {
        return address(this).balance;
    }

    function getClaimableReward(address _device) external view returns (uint256) {
        uint256 totalWeight = contributor.getTotalWeight();
        if (totalWeight == 0) return 0;

        uint256 deviceContribution = contributor.getContribution(_device);
        uint256 rewardAmount = (address(this).balance * deviceContribution) / totalWeight;
        return rewardAmount - rewardsClaimed[_device];
    }

    function getRewardsClaimed(address _device) external view returns (uint256) {
        return lastClaimedReward[_device];
    }
}
