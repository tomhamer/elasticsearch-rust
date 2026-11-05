#!/usr/bin/env python3
"""Extract ES's REST-compatibility test transforms into compat8.json.

ES runs its previous-major YAML suites against the current node
(`yamlRestCompatTest`, every request with `compatible-with=8`) after the
transforms each project lists in its `build.gradle`
(`tasks.named("yamlRestCompatTestTransform") { task -> task.skipTest(...) ... }`).
`es-yaml-runner --compat 8` applies the same transforms; this script reads
them from an ES source checkout.

usage: gen_compat.py <es source dir (9.5.5)> <out compat8.json>

Output: {"projects": {"<project dir>": [{"op": "<task method>", "args": [...]}, ...]},
         "suites": {"oss"|"main"|"sec"|"plug": ["<project dir>", ...]}}
"""
import json, os, re, sys

SRC, OUT = sys.argv[1], sys.argv[2]

# Which projects' transforms apply to which of our suites (see ~/bin/xsetup.sh):
# oss = rest-api-spec + data-streams module (+ x-pack's dlm / data_streams dirs);
# main / sec = x-pack/plugin's own YAML tests; plug = the other x-pack plugins.
SUITES = {
    'oss': ['rest-api-spec', 'modules/data-streams', 'x-pack/plugin'],
    'main': ['x-pack/plugin'],
    'sec': ['x-pack/plugin'],
}


def strip_comments(s):
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c in '"\'':
            j = i + 1
            while j < n and s[j] != c:
                j += 2 if s[j] == '\\' else 1
            out.append(s[i:j + 1])
            i = j + 1
        elif s.startswith('//', i):
            while i < n and s[i] != '\n':
                i += 1
        elif s.startswith('/*', i):
            i = s.find('*/', i) + 2
        else:
            out.append(c)
            i += 1
    return ''.join(out)


def drop_ipv6_block(s):
    # The IPv6-only skips (java.net.preferIPv6Addresses) do not apply.
    m = re.search(r'if\s*\(\s*System\.getProperty\("java\.net\.preferIPv6Addresses"', s)
    while m:
        i = s.index('{', m.end())
        depth, j = 0, i
        while True:
            if s[j] == '{':
                depth += 1
            elif s[j] == '}':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        s = s[:m.start()] + s[j + 1:]
        m = re.search(r'if\s*\(\s*System\.getProperty\("java\.net\.preferIPv6Addresses"', s)
    return s


def parse_args(s, i):
    """Parse a Groovy argument list starting after '('; returns (args, index after ')')."""
    args = []
    n = len(s)
    while i < n:
        c = s[i]
        if c.isspace() or c == ',':
            i += 1
        elif c == ')':
            return args, i + 1
        elif c in '"\'':
            j = i + 1
            buf = []
            while s[j] != c:
                if s[j] == '\\':
                    buf.append(s[j + 1])
                    j += 2
                else:
                    buf.append(s[j])
                    j += 1
            val = ''.join(buf)
            i = j + 1
            # string concatenation with +
            while True:
                k = i
                while k < n and s[k].isspace():
                    k += 1
                if k < n and s[k] == '+':
                    k += 1
                    while s[k].isspace():
                        k += 1
                    q = s[k]
                    j = k + 1
                    buf = []
                    while s[j] != q:
                        if s[j] == '\\':
                            buf.append(s[j + 1])
                            j += 2
                        else:
                            buf.append(s[j])
                            j += 1
                    val += ''.join(buf)
                    i = j + 1
                else:
                    break
            args.append(val)
        else:
            m = re.match(r'(-?\d+(?:\.\d+)?|null|true|false)', s[i:])
            if not m:
                raise ValueError('cannot parse argument at %r' % s[i:i + 80])
            tok = m.group(1)
            args.append(None if tok == 'null' else True if tok == 'true' else False if tok == 'false' else (float(tok) if '.' in tok else int(tok)))
            i += len(tok)
    raise ValueError('unterminated call')


def transforms(path):
    s = strip_comments(open(path).read())
    s = drop_ipv6_block(s)
    ops = []
    for m in re.finditer(r'yamlRestCompatTestTransform', s):
        # the configure block that follows
        i = s.index('{', m.end())
        depth, j = 0, i
        while True:
            if s[j] == '{':
                depth += 1
            elif s[j] == '}':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        block = s[i:j]
        for c in re.finditer(r'\btask\.(\w+)\(', block):
            name = c.group(1)
            if name in ('dependsOn', 'getDestinationDir'):
                continue
            args, _ = parse_args(block, c.end())
            ops.append({'op': name, 'args': args})
    return ops


projects = {}
for root, dirs, files in os.walk(SRC):
    dirs[:] = [d for d in dirs if d not in ('.git', 'build', 'node_modules', 'build-tools', 'build-tools-internal')]
    if 'build.gradle' in files:
        p = os.path.join(root, 'build.gradle')
        ops = transforms(p)
        if ops:
            projects[os.path.relpath(root, SRC)] = ops
suites = dict(SUITES)
suites['plug'] = sorted(p for p in projects if p.startswith('x-pack/plugin/'))
json.dump({'source': 'elasticsearch ' + os.path.basename(os.path.abspath(SRC)), 'projects': projects, 'suites': suites}, open(OUT, 'w'), indent=1, sort_keys=True)
print({k: len(v) for k, v in projects.items()})
