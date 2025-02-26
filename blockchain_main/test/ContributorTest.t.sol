// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.24;

import {Test, console} from "forge-std/Test.sol";
import {Contributor} from "../src/Contributor.sol";
import {DistributeReward} from "../src/DistributeReward.sol";
import {DeployContributor} from "../script/DeployContributor.s.sol";

contract ContributorTest is Test {
    Contributor contributor;

    address USER = makeAddr("user");
    address USER1 = makeAddr("user1");
    uint256 constant STARTING_BALANCE = 10 ether;

    function setUp() external {
        DeployContributor deployContributor = new DeployContributor();
        (contributor,) = deployContributor.run();
        vm.deal(USER, STARTING_BALANCE);
        vm.deal(USER1, STARTING_BALANCE);
    }

    function testSetRewardDistributor() external {
        address rewardDistributorAddress = address(new DistributeReward());
        address currentOwner = contributor.owner();
        vm.prank(currentOwner);
        contributor.setRewardDistributor(rewardDistributorAddress);
        console.log(address(contributor.rewardDistributor()));
        console.log(rewardDistributorAddress);
        assertEq(
            address(contributor.rewardDistributor()), rewardDistributorAddress, "Reward distributor not set correctly"
        );
    }

    function testAddZeroContribution() external {
        vm.prank(USER);
        vm.expectRevert("Contributions must be greater than zero"); // Expecting that it won't affect the contribution
        contributor.addContribution(0, USER);
    }

    function testAddContribution() external {
        uint256 contributionAmount = 5;

        vm.prank(USER);
        contributor.addContribution(contributionAmount, USER);

        assertEq(contributor.getContribution(USER), contributionAmount, "Contribution amount mismatch");
    }

    function testMultipleContribution() external {
        uint256 firstContribution = 3;
        uint256 secondContribution = 6;

        vm.prank(USER);
        contributor.addContribution(firstContribution, USER);
        contributor.addContribution(secondContribution, USER);

        uint256 totalContribution = firstContribution + secondContribution;
        uint256 getContributtions = contributor.getContribution(USER);
        assertEq(getContributtions, totalContribution, "Multiple contributions not added correctly");
    }

    function testTotalWeightUpdate() external {
        uint256 contributionAmount = 5;

        vm.prank(USER);
        contributor.addContribution(contributionAmount, USER);

        assertEq(contributor.getTotalWeight(), contributionAmount, "Contribution amount mismatch");
    }

    function testTotalWeightWithMultipleContributions() external {
        uint256 firstContribution = 3;
        uint256 secondContribution = 6;

        vm.prank(USER);
        contributor.addContribution(firstContribution, USER);
        vm.prank(USER1);
        contributor.addContribution(secondContribution, USER1);

        uint256 totalContribution = firstContribution + secondContribution;
        assertEq(contributor.getTotalWeight(), totalContribution, "Multiple contributions not added correctly");
    }

    function testGetContributionForNonExistentUser() external view {
        uint256 contribution = contributor.getContribution(USER);

        assertEq(contribution, 0, "Non-existent user contribution should be zero");
    }

    function testAddHashContribution() external {
        bytes32 testHash1 = keccak256(abi.encodePacked("test data 1"));
        bytes32 testHash2 = keccak256(abi.encodePacked("test data 2"));

        vm.prank(USER);
        contributor.addHashContribution(USER, testHash1);
        contributor.addHashContribution(USER, testHash2);

        bytes32[] memory storedHases = contributor.getDeviceHashes(USER);
        assertEq(storedHases.length, 2, "Incorrect length of hashes");
        assertEq(storedHases[0], testHash1, "Data 1 mismatch");
        assertEq(storedHases[1], testHash2, "Data 2 mismatch");
    }

    function testGetDeviceHashesForNonExistentUser() external view {
        bytes32[] memory storedHashes = contributor.getDeviceHashes(USER1);
        assertEq(storedHashes.length, 0, "Non-existent user should have no hashes");
    }
}
