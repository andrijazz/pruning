import importlib
import copy
import argparse
import base.factory as factory


def main(args):
    config_module = importlib.import_module('models.basic_config')
    config = copy.deepcopy(config_module.cfg)
    learner = factory.create_learner(config)
    if args.mode == 'train':
        learner.train()
    elif args.mode == 'test':
        learner.test()
    else:
        exit('Unknown mode. Please specify train or test mode')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning Challenge')
    parser.add_argument('--mode', type=str, help='Mode [train | test]', required=True)
    args = parser.parse_args()
    main(args)
