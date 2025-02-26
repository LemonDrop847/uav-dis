-include .env

all: clean remove install update build

clean :; forge clean

remove :; rm -rf .gitmodules && rm -rf .git/modules/* && rm -rf lib && touch .gitmodules && git add . && git commit -m "modules"

install :; forge install OpenZeppelin/openzeppelin-contracts --no-commit

update:; forge update

build:; forge build

test :; forge test

format :; forge fmt

deploy:;
	forge script --rpc-url $(ANVIL_RPC_URL) --private-key $(ANVIL_PRIVATE_KEY) script/DeployContributor.s.sol:DeployContributor --broadcast

deploy-sepolia:
	@forge script script/DeployFundMe.s.sol:DeployFundMe --rpc-url $(SEPOLIA_RPC_URL) --private-key $(PRIVATE_KEY) --broadcast --verify --etherscan-api-key $(ETHERSCAN_API_KEY) -vvvv
