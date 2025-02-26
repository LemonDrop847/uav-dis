// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.24;

import {Script} from "forge-std/Script.sol";
import {Contributor} from "../src/Contributor.sol";
import {DistributeReward} from "../src/DistributeReward.sol";

contract DeployContributor is Script {
    function run() external returns (Contributor, DistributeReward) {
        vm.startBroadcast();
        Contributor contributor = new Contributor();
        DistributeReward rewardDistributor = new DistributeReward();
        contributor.setRewardDistributor(address(rewardDistributor));
        rewardDistributor.setContributor(address(contributor));
        vm.stopBroadcast();
        return (contributor, rewardDistributor);
    }
}
