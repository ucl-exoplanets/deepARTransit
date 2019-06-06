import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '-e', '--experiment',
        metavar='E',
        default='None',
        help='The experiment name')

    args = argparser.parse_args()
    return args

