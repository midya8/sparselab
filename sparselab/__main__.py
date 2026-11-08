"""Strict bounded offline JSON entry points."""
import argparse
import json
import sys
from . import demo, path, screen, screen_demo, branch, branch_demo


def load_input(filename, mode='path'):
    with open(filename, 'rb') as stream:
        data = stream.read(100001)
    if len(data) > 100000:
        raise ValueError('input exceeds 100000 bytes')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    obj = json.loads(data.decode('utf-8'), object_pairs_hook=unique,
                     parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    expected = {'schema_version', 'X', 'y', 'penalties'} if mode == 'path' else {'schema_version', 'X', 'y', 'coefficients', 'penalty'}
    if mode == 'branch':
        expected = {'schema_version', 'X', 'y', 'signs'}
    if not isinstance(obj, dict) or set(obj) != expected:
        raise ValueError('expected exactly ' + ', '.join(sorted(expected)))
    if type(obj['schema_version']) is not int or obj['schema_version'] != 1:
        raise ValueError('schema_version must be integer 1')
    if mode == 'branch':
        return branch(obj['X'], obj['y'], obj['signs'])
    if mode == 'screen':
        return screen(obj['X'], obj['y'], obj['coefficients'], obj['penalty'])
    return path(obj['X'], obj['y'], obj['penalties'])


def main():
    parser = argparse.ArgumentParser(description='Offline LASSO optimality and coefficient ambiguity audits')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('demo')
    commands.add_parser('screen-demo')
    commands.add_parser('branch-demo')
    certificate = commands.add_parser('branch')
    certificate.add_argument('input')
    audit = commands.add_parser('screen')
    audit.add_argument('input')
    solve = commands.add_parser('path')
    solve.add_argument('input')
    args = parser.parse_args()
    try:
        if args.command == 'demo':
            result = demo()
        elif args.command == 'branch-demo':
            result = branch_demo()
        elif args.command == 'screen-demo':
            result = screen_demo()
        else:
            result = load_input(args.input, args.command)
    except (ValueError, OSError, UnicodeError, ArithmeticError) as exc:
        parser.exit(2, 'sparselab: ' + str(exc) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
