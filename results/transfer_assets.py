#!/usr/bin/env python3
"""Print reviewed rsync pulls; --execute opts into transfers from the server.

Run on the receiving machine with Python 3.10+, rsync and a configured SSH alias.
This program does not submit/cancel jobs, upload to GitHub, or delete files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess


HERE = Path(__file__).resolve().parent


def ssh_alias(value):
    if (not value or value.startswith('-')
            or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value)
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value)):
        raise argparse.ArgumentTypeError('Use an SSH alias with letters/digits/dot/underscore/hyphen; no whitespace, control characters or leading hyphen.')
    return value


def command_for(asset, host, root):
    kind = asset.get('kind')
    if kind not in ('file', 'directory'):
        raise ValueError('Asset kind must be file or directory')
    relative = PurePosixPath(asset['destination'])
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Asset destination must stay inside the handoff directory')
    destination = (root / Path(*relative.parts)).resolve()
    destination.relative_to(root)
    source = asset['source_path']
    if (not isinstance(source, str) or not source.startswith('/')
            or any(ord(char) < 32 or ord(char) == 127 for char in source)):
        raise ValueError('Asset source must be an absolute server path without control characters')
    if kind == 'file' and PurePosixPath(source).name != destination.name:
        raise ValueError('Checkpoint filename must be preserved')
    remote = host + ':' + source.rstrip('/') + ('/' if kind == 'directory' else '')
    target = str(destination) + ('/' if kind == 'directory' else '')
    return ['rsync', '-a', '--partial', '--protect-args', '--', remote, target], destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--list', action='store_true', help='List asset IDs and destinations; no connection or directory creation.')
    parser.add_argument('--asset', action='append', default=[], metavar='ID',
                        help='Asset ID from asset_manifest.json; repeat to pull multiple assets.')
    parser.add_argument('--host', type=ssh_alias, help='Configured SSH alias, e.g. deltaai.')
    parser.add_argument('--destination-root', type=Path,
                        help='Local handoff_20261103 directory, not its parent repository.')
    parser.add_argument('--execute', action='store_true', help='Run the printed transfers; omitted means print only.')
    args = parser.parse_args(argv)
    manifest = json.loads((HERE / 'asset_manifest.json').read_text())
    assets = {}
    for item in manifest['assets']:
        if item['id'] in assets:
            parser.error('Duplicate asset ID in manifest: ' + item['id'])
        assets[item['id']] = item
    if args.list:
        for item in assets.values():
            print(f"{item['id']}\t{item['kind']}\t{item.get('size_bytes')} bytes\t{item['destination']}")
        if not args.asset and not args.execute:
            return
    if not args.asset or args.host is None or args.destination_root is None:
        parser.error('A pull requires --asset ID --host SSH_ALIAS --destination-root HANDOFF; use --list to inspect IDs.')
    root = args.destination_root.expanduser().resolve()
    requests = []
    for identifier in dict.fromkeys(args.asset):
        if identifier not in assets:
            parser.error('Unknown asset ID: ' + identifier)
        try:
            command, target = command_for(assets[identifier], args.host, root)
        except (ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
        requests.append((assets[identifier], command, target))
    if args.execute and shutil.which('rsync') is None:
        parser.error('rsync is not installed; run from a receiver environment such as WSL with rsync and SSH configured.')
    for item, command, target in requests:
        print(f"{item['id']}: {item.get('size_bytes')} bytes; {item['destination']}", flush=True)
        if item.get('note'):
            print(item['note'], flush=True)
        print(shlex.join(command), flush=True)
        if args.execute:
            if item['kind'] == 'directory':
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True)
    if not args.execute:
        print('Print only. Add --execute to start these pulls. No directories or transfers were created.')


if __name__ == '__main__':
    main()
