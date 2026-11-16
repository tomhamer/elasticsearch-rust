#!/usr/bin/env python3
"""Translate one of Lucene's hand-written `char[] s, int len` stemmers
(GreekStemmer, CzechStemmer, ...) into Rust over the helpers of
`rustic_core::analysis::stemmer::lucene` (`ends_with`, `starts_with`,
`delete_n`, `contains`). The subset handled is the one these classes use:
int / boolean methods over (char[], int), fields, static string sets,
if / else (with or without braces), switch on a char or int with
fall-through case groups, returns, compound assignments, ++ / --.

usage: lucene_stemmer2rs.py <Stemmer.java> <RustStructName> > out.rs
"""
import re, sys

src = open(sys.argv[1]).read()
name = sys.argv[2]

# strip comments
src = re.sub(r'/\*.*?\*/', '', src, flags=re.S)
src = re.sub(r'//[^\n]*', '', src)

TOK = re.compile(r'\s*(?:(\d+)|("(?:[^"\\]|\\.)*")|(\'(?:[^\'\\]|\\.)*\')|([A-Za-z_]\w*)|(\+\+|--|\+=|-=|\|=|&=|==|!=|<=|>=|&&|\|\||[{}()\[\];,.=<>!+\-*/%&|?:]))')

def tokenize(s):
    out = []
    pos = 0
    while pos < len(s):
        m = TOK.match(s, pos)
        if not m:
            if s[pos:].strip() == '':
                break
            raise SystemExit('cannot tokenize at %r' % s[pos:pos + 40])
        pos = m.end()
        out.append(next(g for g in m.groups() if g is not None))
    return out

toks = tokenize(src)
i0 = toks.index('class')
# class body
assert toks[i0 + 1] == name or True
start = toks.index('{', i0) + 1

def find_close(t, i, o='{', c='}'):
    d = 1
    while d:
        if t[i] == o: d += 1
        elif t[i] == c: d -= 1
        i += 1
    return i  # index after the close

end = find_close(toks, start) - 1
body = toks[start:end]

sets = {}      # NAME -> list of strings
fields = []    # (name, rust type)
consts = []    # (name, rust type, value)
methods = []   # (name, ret, params, body tokens)
i = 0
JTYPE = {'int': 'i32', 'boolean': 'bool', 'char': 'char'}
while i < len(body):
    j = i
    while body[j] in ('private', 'static', 'final', 'public', 'protected'):
        j += 1
    if body[j] == 'CharArraySet':
        nm = body[j + 1]
        k = body.index(';', j)
        strs = [t[1:-1] for t in body[j:k] if t.startswith('"')]
        sets[nm] = strs
        i = k + 1
        continue
    if body[j] in JTYPE and body[j + 2] == '(':
        ret, nm = body[j], body[j + 1]
        k = find_close(body, j + 3, '(', ')')
        params = body[j + 3:k - 1]
        assert body[k] == '{', (nm, body[k])
        e = find_close(body, k + 1)
        if nm not in ('endsWith', 'startsWith'):  # the shared helpers
            methods.append((nm, ret, params, body[k + 1:e - 1]))
        i = e
        continue
    if body[j] in JTYPE and body[j + 2] in (';', '='):
        k = body.index(';', j)
        if 'static' in body[i:j] and 'final' in body[i:j]:
            consts.append((body[j + 1], JTYPE[body[j]], ' '.join(body[j + 3:k])))
        else:
            fields.append((body[j + 1], JTYPE[body[j]]))
        i = k + 1
        continue
    raise SystemExit('unhandled class member near %r' % body[j:j + 8])

def rust_str(t):
    s = t[1:-1]
    s = re.sub(r'\\u([0-9A-Fa-f]{4})', lambda m: '\\u{%s}' % m.group(1), s)
    return '"' + s + '"'

def rust_char(t):
    s = t[1:-1]
    s = re.sub(r'\\u([0-9A-Fa-f]{4})', lambda m: '\\u{%s}' % m.group(1), s)
    return "'" + s + "'"

field_names = {f for f, _ in fields}
method_names = {m[0] for m in methods}
arrays = set()

def expr(ts):
    """Translate an expression token list."""
    out = []
    i = 0
    while i < len(ts):
        t = ts[i]
        if t.startswith('"'):
            out.append(rust_str(t))
        elif t.startswith("'"):
            out.append(rust_char(t))
        elif t in sets and i + 2 < len(ts) and ts[i + 1] == '.' and ts[i + 2] == 'contains':
            k = find_close(ts, i + 4, '(', ')')
            args = split_args(ts[i + 4:k - 1])
            # contains(s, 0, len)
            out.append('contains(%s, %s, %s)' % (t.upper(), expr(args[0]), expr(args[2])))
            i = k
            continue
        elif t in ('endsWith', 'startsWith') and ts[i + 1] == '(':
            k = find_close(ts, i + 2, '(', ')')
            args = split_args(ts[i + 2:k - 1])
            fn = 'ends_with' if t == 'endsWith' else 'starts_with'
            out.append('%s(%s, %s, %s)' % (fn, expr(args[0]), expr(args[1]), expr(args[2])))
            i = k
            continue
        elif t == 'deleteN' and ts[i + 1] == '(':
            k = find_close(ts, i + 2, '(', ')')
            args = split_args(ts[i + 2:k - 1])
            out.append('delete_n(%s)' % ', '.join(expr(a) for a in args))
            i = k
            continue
        elif t in method_names and i + 1 < len(ts) and ts[i + 1] == '(':
            k = find_close(ts, i + 2, '(', ')')
            args = split_args(ts[i + 2:k - 1]) if k - 1 > i + 2 else []
            out.append('self.%s(%s)' % (snake(t), ', '.join(expr(a) for a in args)))
            i = k
            continue
        elif t in field_names:
            out.append('self.%s' % snake(t))
        elif re.match(r'[A-Za-z_]\w*$', t) and i + 1 < len(ts) and ts[i + 1] == '[':
            k = find_close(ts, i + 2, '[', ']')
            out.append('%s[(%s) as usize]' % (t, expr(ts[i + 2:k - 1])))
            i = k
            continue
        elif t == '.' and i + 1 < len(ts) and ts[i + 1] == 'length':
            out.append('.len() as i32')
            i += 2
            continue
        elif re.match(r'[A-Za-z_]\w*$', t):
            out.append(snake(t))
        else:
            out.append(t)
        i += 1
    s = ' '.join(out)
    s = s.replace('( ', '(').replace(' )', ')').replace(' ,', ',').replace(' .', '.').replace('. ', '.')
    return s

def snake(n):
    if n in ('s', 'len', 'buffer', 'text', 'length', 'true', 'false'):
        return n
    return re.sub(r'(?<=[a-z0-9])([A-Z])', lambda m: '_' + m.group(1).lower(), n)

def split_args(ts):
    args, cur, d = [], [], 0
    for t in ts:
        if t in '([': d += 1
        if t in ')]': d -= 1
        if t == ',' and d == 0:
            args.append(cur); cur = []
        else:
            cur.append(t)
    if cur: args.append(cur)
    return args

def stmt(ts, i, ind, out):
    """Translate the statement at ts[i]; return the next index."""
    t = ts[i]
    if t == '{':
        e = find_close(ts, i + 1)
        out.append(ind + '{')
        block(ts[i + 1:e - 1], ind + '    ', out)
        out.append(ind + '}')
        return e
    if t == 'if':
        k = find_close(ts, i + 2, '(', ')')
        cond = expr(ts[i + 2:k - 1])
        out.append(ind + 'if %s {' % cond)
        n = stmt_inner(ts, k, ind + '    ', out)
        if n < len(ts) and ts[n] == 'else':
            if ts[n + 1] == 'if':
                out.append(ind + '} else')
                return stmt(ts, n + 1, ind, out) if False else else_if(ts, n + 1, ind, out)
            out.append(ind + '} else {')
            n = stmt_inner(ts, n + 1, ind + '    ', out)
        out.append(ind + '}')
        return n
    if t == 'switch':
        k = find_close(ts, i + 2, '(', ')')
        scrut = expr(ts[i + 2:k - 1])
        assert ts[k] == '{'
        e = find_close(ts, k + 1)
        sw = ts[k + 1:e - 1]
        out.append(ind + 'match %s {' % scrut)
        j = 0
        groups = []
        while j < len(sw):
            labels = []
            while j < len(sw) and sw[j] in ('case', 'default'):
                if sw[j] == 'default':
                    labels.append('_'); j += 2
                else:
                    lab = sw[j + 1]
                    labels.append(rust_char(lab) if lab.startswith("'") else lab); j += 3
            body_start = j
            while j < len(sw) and sw[j] not in ('case', 'default'):
                if sw[j] == '{':
                    j = find_close(sw, j + 1); continue
                j += 1
            groups.append((labels, sw[body_start:j]))
        # fall-through: a group without statements joins the next one
        merged = []
        pend = []
        for labels, b in groups:
            pend += labels
            if b:
                merged.append((pend, b)); pend = []
        if pend:
            merged.append((pend, []))
        has_default = False
        for labels, b in merged:
            if '_' in labels:
                has_default = True
                labels = ['_']
            out.append(ind + '    %s => {' % ' | '.join(labels))
            bb = [x for x in b]
            if bb and bb[-2:] == ['break', ';']:
                bb = bb[:-2]
            block(bb, ind + '        ', out)
            out.append(ind + '    }')
        if not has_default:
            out.append(ind + '    _ => {}')
        out.append(ind + '}')
        return e
    if t == 'return':
        k = ts.index(';', i)
        out.append(ind + 'return %s;' % expr(ts[i + 1:k]))
        return k + 1
    if t == 'for':
        k = find_close(ts, i + 2, '(', ')')
        hdr = ts[i + 2:k - 1]
        parts = split_semis(hdr)
        init, cond, step = parts
        # for (int i = 0; i < len; i++)
        var = init[1] if init[0] == 'int' else init[0]
        lo = expr(init[3:] if init[0] == 'int' else init[2:])
        out.append(ind + 'let mut %s: i32 = %s;' % (var, lo))
        out.append(ind + 'while %s {' % expr(cond))
        n = stmt_inner(ts, k, ind + '    ', out)
        out.append(ind + '    %s' % simple_stmt(step))
        out.append(ind + '}')
        return n
    if t == 'break':
        out.append(ind + 'break;')
        return i + 2
    # declarations
    j = i
    while ts[j] == 'final':
        j += 1
    if ts[j] in JTYPE and re.match(r'[A-Za-z_]\w*$', ts[j + 1]) and ts[j + 2] in ('=', ';'):
        k = ts.index(';', j)
        if ts[j + 2] == '=':
            out.append(ind + 'let mut %s: %s = %s;' % (snake(ts[j + 1]), JTYPE[ts[j]], expr(ts[j + 3:k])))
        else:
            out.append(ind + 'let mut %s: %s;' % (snake(ts[j + 1]), JTYPE[ts[j]]))
        return k + 1
    if ts[j] == 'Affix' and ts[j + 2] == '=':
        k = ts.index(';', j)
        out.append(ind + 'let %s = &%s;' % (snake(ts[j + 1]), expr(ts[j + 3:k]).replace('affixes', 'AFFIXES')))
        return k + 1
    k = semi(ts, i)
    out.append(ind + simple_stmt(ts[i:k]))
    return k + 1

def else_if(ts, i, ind, out):
    # emits "if ... {" continuing a previous "} else" line
    k = find_close(ts, i + 2, '(', ')')
    out[-1] = out[-1] + ' if %s {' % expr(ts[i + 2:k - 1])
    n = stmt_inner(ts, k, ind + '    ', out)
    if n < len(ts) and ts[n] == 'else':
        if ts[n + 1] == 'if':
            out.append(ind + '} else')
            return else_if(ts, n + 1, ind, out)
        out.append(ind + '} else {')
        n = stmt_inner(ts, n + 1, ind + '    ', out)
    out.append(ind + '}')
    return n

def stmt_inner(ts, i, ind, out):
    if ts[i] == '{':
        e = find_close(ts, i + 1)
        block(ts[i + 1:e - 1], ind, out)
        return e
    return stmt(ts, i, ind, out)

def semi(ts, i):
    d = 0
    while True:
        if ts[i] in '([': d += 1
        if ts[i] in ')]': d -= 1
        if ts[i] == ';' and d == 0:
            return i
        i += 1

def split_semis(ts):
    parts, cur, d = [], [], 0
    for t in ts:
        if t in '([': d += 1
        if t in ')]': d -= 1
        if t == ';' and d == 0:
            parts.append(cur); cur = []
        else:
            cur.append(t)
    parts.append(cur)
    return parts

def simple_stmt(ts):
    if len(ts) == 2 and ts[1] in ('++', '--'):
        return '%s %s 1;' % (expr(ts[:1]), '+=' if ts[1] == '++' else '-=')
    return expr(ts) + ';'

def block(ts, ind, out):
    i = 0
    while i < len(ts):
        i = stmt(ts, i, ind, out)

out = []
out.append('// Translated from Lucene 10.3.2 `%s` by tools/lucene_stemmer2rs.py.' % name)
out.append('#![allow(unused, unused_parens, unused_mut, non_snake_case, clippy::all)]')
out.append('use super::lucene::{contains, delete_n, ends_with, starts_with};')
out.append('')
for nm, strs in sets.items():
    out.append('static %s: &[&str] = &[%s];' % (nm.upper(), ', '.join('"%s"' % s for s in strs)))
for nm, t, v in consts:
    out.append('const %s: %s = %s;' % (nm, t, v))
out.append('')
out.append('#[derive(Default)]')
out.append('pub struct %s {' % name)
for f, t in fields:
    out.append('    %s: %s,' % (snake(f), t))
out.append('}')
out.append('')
out.append('impl %s {' % name)
for nm, ret, params, b in methods:
    ps = split_args(params)
    rparams = []
    for p in ps:
        p = [x for x in p if x != 'final']
        if p[0] == 'char' and p[1] == '[':
            rparams.append('%s: &mut Vec<char>' % p[3])
        elif p[0] == 'char' and len(p) == 4 and p[2] == '[':
            rparams.append('%s: &mut Vec<char>' % p[1])
        else:
            rparams.append('mut %s: %s' % (snake(p[1]), JTYPE[p[0]]))
    out.append('    pub fn %s(&mut self, %s) -> %s {' % (snake(nm), ', '.join(rparams), JTYPE[ret]))
    block(b, '        ', out)
    out.append('    }')
out.append('}')
print('\n'.join(out))
