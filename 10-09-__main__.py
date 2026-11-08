"""Strict bounded offline JSON entry points."""
import argparse
import json
import sys
from . import demo, path


def load_input(filename):
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
    if not isinstance(obj, dict) or set(obj) != {'schema_version', 'X', 'y', 'penalties'}:
        raise ValueError('expected exactly schema_version, X, y, penalties')
    if type(obj['schema_version']) is not int or obj['schema_version'] != 1:
        raise ValueError('schema_version must be integer 1')
    return path(obj['X'], obj['y'], obj['penalties'])


def main():
    parser = argparse.ArgumentParser(description='Offline LASSO optimality and coefficient ambiguity audits')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('demo')
    solve = commands.add_parser('path')
    solve.add_argument('input')
    args = parser.parse_args()
    try:
        result = demo() if args.command == 'demo' else load_input(args.input)
    except (ValueError, OSError, UnicodeError, ArithmeticError) as exc:
        parser.exit(2, 'sparselab: ' + str(exc) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
