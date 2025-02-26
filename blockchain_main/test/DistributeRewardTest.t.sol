// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.24;

import {Test, console} from "forge-std/Test.sol";
import {Contributor} from "../src/Contributor.sol";
import {DistributeReward} from "../src/DistributeReward.sol";
import {DeployContributor} from "../script/DeployContributor.s.sol";

contract DistributeRewardTest is Test {
    Contributor contributor;
    DistributeReward distributeReward;

    address USER = makeAddr("user");
    address USER1 = makeAddr("user1");
    uint256 constant STARTING_BALANCE = 10 ether;

    function setUp() external {
        DeployContributor deployContributor = new DeployContributor();
        (contributor, distributeReward) = deployContributor.run();

        vm.deal(address(distributeReward), 5 ether);
        vm.deal(USER, STARTING_BALANCE);
        vm.deal(USER1, STARTING_BALANCE);
        console.log("contributor address", address(contributor));
        console.log("distributor address", address(distributeReward));
    }

    function testDepositReward() external {
        uint256 depositAmount = 8 ether;
        vm.prank(USER);
        distributeReward.depositReward{value: depositAmount}();
        uint256 rewardPoolBalance = distributeReward.getRewardPoolBalance();
        assertEq(rewardPoolBalance, depositAmount + 5 ether, "Reward pool balance mismatch");
    }

    function testDistributeReward() external {
        uint256 contributionAmount = 5 ether;

        // Add contributions
        vm.prank(USER);
        contributor.addContribution(contributionAmount, USER);

        // Deposit rewards
        uint256 depositAmount = 5 ether;
        vm.prank(USER);
        distributeReward.depositReward{value: depositAmount}();

        // Calculate rewards
        distributeReward.calculateReward();

        // Distribute rewards to USER
        vm.prank(USER);
        uint256 rewardBefore = distributeReward.getClaimableReward(USER);
        distributeReward.distributeReward(USER);

        // Check that USER received the reward
        uint256 rewardAfter = distributeReward.getClaimableReward(USER);
        assertEq(rewardAfter, rewardBefore - rewardBefore, "Reward amount mismatch after distribution");
    }

    function testDistributeRewardForNonExistentUser() external {
        // Deposit rewards
        vm.prank(USER);
        distributeReward.depositReward{value: 10 ether}();

        // Add contribution for USER
        uint256 contributionAmount = 7 ether;
        vm.prank(USER);
        contributor.addContribution(contributionAmount, USER);

        // Expect revert for non-existent user USER1
        distributeReward.calculateReward();
        vm.expectRevert("No contribution from device");
        distributeReward.distributeReward(USER1);
    }

    function testCalculateRewards() external {
        uint256 contributionAmountUser = 3 ether;
        uint256 contributionAmountUser1 = 2 ether;

        // Add contributions
        vm.prank(USER);
        contributor.addContribution(contributionAmountUser, USER);
        vm.prank(USER1);
        contributor.addContribution(contributionAmountUser1, USER1);

        // Deposit rewards
        uint256 depositAmount = 5 ether;
        vm.prank(USER);
        distributeReward.depositReward{value: depositAmount}();

        // Calculate rewards
        distributeReward.calculateReward();

        // Check reward addresses for both users
        uint256 rewardForUser = distributeReward.rewardAddress(USER);
        uint256 rewardForUser1 = distributeReward.rewardAddress(USER1);

        uint256 expectedRewardUser =
            ((depositAmount + 5 ether) * contributionAmountUser) / (contributionAmountUser + contributionAmountUser1);
        uint256 expectedRewardUser1 =
            ((depositAmount + 5 ether) * contributionAmountUser1) / (contributionAmountUser + contributionAmountUser1);

        assertEq(rewardForUser, expectedRewardUser, "Reward for USER is incorrect");
        assertEq(rewardForUser1, expectedRewardUser1, "Reward for USER1 is incorrect");
    }
}
