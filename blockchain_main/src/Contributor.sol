// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.24;

import "./DistributeReward.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Contributor is Ownable {
    uint256 public totalWeight;
    mapping(address => uint256) public devices;
    mapping(address => bytes32[]) public deviceHashes;
    DistributeReward public rewardDistributor;

    address[] public deviceList;
    mapping(address => bool) public isDeviceAdded;

    event ContributionAdded(address indexed device, uint256 weight);
    event TotalWeightUpdated(uint256 newTotalWeight);
    event RewardDistributorSet(address rewardDistributor);
    event RewardDistributed(address indexed device, uint256 rewardAmount);
    event HashContributionAdded(address indexed device, bytes32 hash);

    constructor() Ownable(msg.sender) {
        totalWeight = 0;
    }

    function setRewardDistributor(address _rewardDistributor) external onlyOwner {
        rewardDistributor = DistributeReward(_rewardDistributor);
        emit RewardDistributorSet(_rewardDistributor);
    }

    function addContribution(uint256 _weight, address _device) public {
        require(_weight > 0, "Contributions must be greater than zero");

        if (!isDeviceAdded[_device]) {
            deviceList.push(_device);
            isDeviceAdded[_device] = true;
        }

        devices[_device] += _weight;
        totalWeight += _weight;

        emit ContributionAdded(_device, _weight);
        emit TotalWeightUpdated(totalWeight);
    }

    function addHashContribution(address _device, bytes32 _hash) public {
        deviceHashes[_device].push(_hash);
        emit HashContributionAdded(_device, _hash);
    }

    function getDeviceList() public view returns (address[] memory) {
        return deviceList;
    }

    function getDeviceHashes(address _device) public view returns (bytes32[] memory) {
        return deviceHashes[_device];
    }

    function getTotalWeight() public view returns (uint256) {
        return totalWeight;
    }

    function getContribution(address _device) public view returns (uint256) {
        return devices[_device];
    }
}
