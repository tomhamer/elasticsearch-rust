#!/usr/bin/env python3
"""Materialize complete immutable fixture inventories on an idle EC2 host only."""
import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import stat
import subprocess
from consolidated_correctness import idle, pinned, sha

COUNTS = {'oss': 571, 'main': 254, 'sec': 74, 'plug': 305}

def readable_ec2_identity():
    """Unreadable DMI fields are not evidence; require a readable EC2 identity."""
    for name in ('/sys/hypervisor/uuid', '/sys/devices/virtual/dmi/id/product_uuid',
                 '/sys/devices/virtual/dmi/id/board_asset_tag'):
        try:
            identity = pathlib.Path(name).read_text().strip().lower()
        except (OSError, UnicodeError):
            continue
        if identity.startswith('ec2') or identity.startswith('i-'):
            return True
    return False


def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def relative(name):
    p = pathlib.PurePosixPath(name)
    require(isinstance(name, str) and str(p) == name and not p.is_absolute()
            and '..' not in p.parts and name not in ('', '.'), 'Unsafe inventory path')
    return p

def snapshot(root):
    rows = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = pathlib.Path(directory) / name
            info = path.lstat()
            kind = stat.S_IFMT(info.st_mode)
            require(kind in (stat.S_IFDIR, stat.S_IFREG, stat.S_IFLNK), 'Special fixture file')
            link = os.readlink(path) if kind == stat.S_IFLNK else None
            if link is not None:
                resolved = path.resolve(strict=True)
                require(resolved.is_relative_to(root) and resolved.is_file(), 'Escaping or directory fixture link')
            rows[str(path.relative_to(root))] = (kind, info.st_dev, info.st_ino, info.st_size,
                                                info.st_mtime_ns, info.st_ctime_ns, link)
    return rows

def hashed(root, rows):
    return {name: sha(root / name) for name, row in sorted(rows.items()) if row[0] != stat.S_IFDIR}

def write_new(path, value):
    with pathlib.Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

def main():
    require(__debug__, 'Assertions must remain enabled for pinned helper checks')
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--yaml-files', required=True)
    p.add_argument('--yaml-files-sha256', required=True)
    p.add_argument('--yaml-provenance-sha256', required=True)
    p.add_argument('--es-manifest', required=True)
    p.add_argument('--es-manifest-sha256', required=True)
    p.add_argument('--resolved-config', required=True)
    p.add_argument('--proof', required=True)
    a = p.parse_args()
    require(os.uname().sysname == 'Linux' and readable_ec2_identity(), 'EC2 only')
    cfg = json.loads(pathlib.Path(a.config).read_text())
    paths = [pathlib.Path(cfg[k]).absolute() for k in ('yaml_inventory', 'es_inventory')]
    paths += [pathlib.Path(a.resolved_config).absolute(), pathlib.Path(a.proof).absolute()]
    require(len({str(x.resolve()) for x in paths}) == len(paths), 'Duplicate output paths')
    require(all(not x.exists() and not x.is_symlink() and x.parent.is_dir() for x in paths), 'Output exists or parent missing')
    roots = [pathlib.Path(cfg[k]).resolve(strict=True) for k in ('yaml_root', 'es_distribution')]
    require(all(not x.resolve().is_relative_to(root) for x in paths for root in roots), 'Outputs must be outside fixture roots')
    frozen_source = (pathlib.Path(cfg['root']) / 'source').resolve(strict=True)
    require(all(not x.resolve().is_relative_to(frozen_source) for x in paths), 'Outputs must be outside immutable source')
    service = subprocess.check_output(['systemctl', 'show', cfg['prerequisite_service'], '--property=Id', '--property=ActiveState'], text=True)
    properties = dict(x.split('=', 1) for x in service.splitlines() if '=' in x)
    expected_service = cfg['prerequisite_service'].removesuffix('.service') + '.service'
    require(expected_service == pathlib.Path(cfg['root']).name + '.service', 'Prerequisite service/results root differ')
    require(properties.get('Id') == expected_service and properties.get('ActiveState') in ('inactive', 'failed'), 'Wrong or active suite service')
    exit_code = int((pathlib.Path(cfg['root']) / 'results/exit.code').read_text())
    pinned(pathlib.Path(cfg['root']) / 'results/elasticsearch-node', cfg['candidate_sha256'])
    pinned(a.yaml_files, a.yaml_files_sha256)
    pinned(a.es_manifest, a.es_manifest_sha256)
    expected = json.loads(pathlib.Path(a.yaml_files).read_text())
    require(set(expected) == set(COUNTS) and all(len(expected[k]) == COUNTS[k] and len(set(expected[k])) == COUNTS[k] for k in COUNTS), 'Original YAML module counts differ')
    for names in expected.values():
        for name in names: relative(name)
    es_expected = json.loads(pathlib.Path(a.es_manifest).read_text())
    require(es_expected['version'] == '9.5.5' and pathlib.Path(es_expected['root']).resolve() == roots[1], 'Wrong ES distribution')
    for name in es_expected['files']: relative(name)
    with open('/tmp/es-pr3-validation-host.lock', 'a+') as host, open(pathlib.Path(cfg['target']) / '.pr3-build.lock', 'a+') as target:
        fcntl.flock(host, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(target, fcntl.LOCK_EX | fcntl.LOCK_NB)
        idle()
        before = [snapshot(root) for root in roots]
        yaml_hashes, es_hashes = [hashed(root, rows) for root, rows in zip(roots, before)]
        pinned(roots[0] / 'provenance.json', a.yaml_provenance_sha256)
        provenance = json.loads((roots[0] / 'provenance.json').read_text())
        original = {}
        for row in provenance['files']:
            name = str(relative(row['target']))
            require(name not in original or original[name] == row['sha256'], 'Conflicting original YAML copy')
            original[name] = row['sha256']
        corpus = {k: v for k, v in yaml_hashes.items() if len(pathlib.PurePosixPath(k).parts) >= 3 and pathlib.PurePosixPath(k).parts[0] in COUNTS and pathlib.PurePosixPath(k).parts[1] in ('api', 'test')}
        require(corpus == original, 'YAML content/path set differs from original materialization provenance')
        excluded = {}
        for suite in COUNTS:
            require(all((roots[0] / suite / kind).is_dir() for kind in ('api', 'test')), 'Missing module directory')
            require(any(k.startswith(suite + '/api/') for k in corpus), 'Empty module API corpus')
            actual = {k[len(suite + '/test/'):] for k in corpus if k.startswith(suite + '/test/') and k.endswith('.yml')}
            omitted = {k for k in actual if suite == 'plug' and k.startswith(('watcher/mustache/', 'watcher/painless/'))}
            require(actual - omitted == set(expected[suite]), 'YAML selected file set differs')
            excluded[suite] = sorted(omitted)
        require(set(es_hashes) == set(es_expected['files']), 'ES distribution file set differs')
        for name, entry in es_expected['files'].items():
            info = before[1][name]
            if 'symlink' in entry:
                require(info[0] == stat.S_IFLNK and info[6] == entry['symlink'], 'ES symlink topology differs')
            else:
                require(info[0] == stat.S_IFREG and info[3] == entry['size'] and es_hashes[name] == entry['sha256'], 'ES original file differs')
        require(all(snapshot(root) == rows for root, rows in zip(roots, before)), 'Fixture changed while hashing')
        idle()
        write_new(paths[0], yaml_hashes)
        write_new(paths[1], es_hashes)
        cfg['yaml_inventory_sha256'] = sha(paths[0])
        cfg['es_inventory_sha256'] = sha(paths[1])
        write_new(paths[3], {'complete': True, 'suite_exit': exit_code, 'yaml_original_sha256': a.yaml_files_sha256,
            'yaml_provenance_sha256': a.yaml_provenance_sha256, 'es_original_sha256': a.es_manifest_sha256,
            'yaml_files': len(yaml_hashes), 'es_files': len(es_hashes), 'selected_suite_counts': COUNTS,
            'retained_watcher_exclusions_not_passes': excluded, 'yaml_inventory_sha256': cfg['yaml_inventory_sha256'],
            'es_inventory_sha256': cfg['es_inventory_sha256'], 'topology': [{k: {'kind': v[0], 'link': v[6]} for k, v in rows.items()} for rows in before]})
        write_new(paths[2], cfg)

if __name__ == '__main__':
    main()
