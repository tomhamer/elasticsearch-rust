#!/usr/bin/env python3
"""AWS-only ordinary two-node fixture matching EnterpriseSearchRestIT's realms.
Run after building on AWS. Ports 19840/19841, transport 19940/19941.
"""
import base64
import hashlib
import json
import pathlib
import re
import secrets
import subprocess
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '/home/ubuntu/es9-feature-work')
fixture = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else '/tmp/es9-entsearch-product')
conf = fixture / 'conf'
conf.mkdir(parents=True, exist_ok=True)
(fixture / 'repo').mkdir(exist_ok=True)
source = (root / 'crates/es-yaml-runner/src/exec.rs').read_text()
section = source.split('fn entsearch_security_config()', 1)[1].split('\nfn ', 1)[0]
roles_text = re.search(r'let roles = serde_json::json!\((.*?)\);', section, re.S).group(1)
roles = json.loads(re.sub(r',\s*([}\]])', r'\1', roles_text))
# Reuse the pinned suite's realm definitions; the stable file administrator
# handles test cleanup independently of the module's default client role.
users = {'entsearch-superuser': 'superuser', 'entsearch-admin': 'admin',
         'entsearch-user': 'user', 'entsearch-unprivileged': 'unprivileged',
         'x_pack_rest_user': 'superuser'}
def hashed(password):
    salt = secrets.token_bytes(32)
    key = hashlib.pbkdf2_hmac('sha512', password.encode(), salt, 1000, 32)
    return '{PBKDF2}1000$' + base64.b64encode(salt).decode() + '$' + base64.b64encode(key).decode()
(conf / 'users').write_text(''.join(f'{user}:{hashed("x-pack-test-password" if user == "x_pack_rest_user" else user + "-password")}\n' for user in users))
by_role = {}
for user, role in users.items():
    by_role.setdefault(role, []).append(user)
(conf / 'users_roles').write_text(''.join(f'{role}:{",".join(names)}\n' for role, names in by_role.items()))
(conf / 'roles.yml').write_text(json.dumps(roles, indent=2))
import os
env = dict(os.environ, ES_PATH_CONF=str(conf))
# `bootstrap.password` is a secure setting (refused in elasticsearch.yml /
# -E): it goes into the shared keystore before the nodes start.
tool = root / 'target/fast/elasticsearch-keystore'
secure = {}
if tool.exists():
    if not (conf / 'elasticsearch.keystore').exists():
        subprocess.run([str(tool), 'create'], env=env, stdin=subprocess.DEVNULL, check=True, capture_output=True)
    subprocess.run([str(tool), 'add', '-x', '-f', 'bootstrap.password'], env=env, input='x-pack-test-password\n',
                   text=True, check=True, capture_output=True)
elif os.environ.get('ES_FIXTURE_PLAIN_SECURE_SETTINGS') == '1':
    # Only for a build older than the keystore.
    secure['bootstrap.password'] = 'x-pack-test-password'
else:
    sys.exit(f'{tool} is missing: build it (cargo build -p elasticsearch --bin elasticsearch-keystore), '
             'or set ES_FIXTURE_PLAIN_SECURE_SETTINGS=1 for a build older than the keystore')
pids = []
for n in (0, 1):
    opts = {'cluster.name': 'entsearch-parity', 'node.name': f'ent-{n}',
            'path.data': str(fixture / f'node-{n}'), 'path.repo': str(fixture / 'repo'),
            'network.host': '127.0.0.1', 'http.port': str(19840+n), 'transport.port': str(19940+n),
            'discovery.seed_hosts': '127.0.0.1:19940,127.0.0.1:19941',
            'cluster.initial_master_nodes': 'ent-0,ent-1',
            'xpack.security.enabled': 'true', 'xpack.security.authc.api_key.enabled': 'true',
            'xpack.license.self_generated.type': 'trial', **secure,
            'cluster.routing.allocation.disk.watermark.low': '1b',
            'cluster.routing.allocation.disk.watermark.high': '1b',
            'cluster.routing.allocation.disk.watermark.flood_stage': '1b',
            'action.destructive_requires_name': 'false'}
    args = [str(root / 'target/fast/elasticsearch')]
    for key, value in opts.items():
        args.extend(['-E', f'{key}={value}'])
    proc = subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL,
                            stdout=open(fixture / f'node-{n}.log', 'w'),
                            stderr=subprocess.STDOUT, start_new_session=True)
    pids.append(proc.pid)
(fixture / 'pids.json').write_text(json.dumps(pids))
print('Enterprise Search fixture PIDs:', pids)
