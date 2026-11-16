"""Regenerate src/locale_names.tsv from the cldr-dates-full npm package.

Usage: unpack cldr-dates-full-46.0.0.tgz into /tmp/cldr (package/main/<locale>/ca-gregorian.json),
then `python3 tools/locale_names_tsv.py src/locale_names.tsv /tmp/new.tsv` (the first argument is the
current table, compared column by column; the second the output).

Columns: locale, months (format wide, format abbreviated, stand-alone wide, stand-alone abbreviated),
weekdays Monday first (same four forms), AM|PM (format wide: what the JDK's `a` prints). A locale whose
rows equal its nearest listed ancestor (dropping subtags) is omitted.
"""
import json, os, sys
base='/tmp/cldr/package/main'
old={}
for line in open(sys.argv[1], encoding='utf-8'):
    f=line.rstrip('\n').split('\t'); old[f[0]]=f[1:]
def rows(loc):
    g=json.load(open(f'{base}/{loc}/ca-gregorian.json', encoding='utf-8'))['main'][loc]['dates']['calendars']['gregorian']
    m=g['months']; d=g['days']; p=g['dayPeriods']['format']['wide']
    order=['mon','tue','wed','thu','fri','sat','sun']
    lst=[]
    for ctx in ['format','stand-alone']:
        for w in ['wide','abbreviated']:
            lst.append('|'.join(m[ctx][w][str(i)] for i in range(1,13)))
    for ctx in ['format','stand-alone']:
        for w in ['wide','abbreviated']:
            lst.append('|'.join(d[ctx][w][k] for k in order))
    lst.append(p['am']+'|'+p['pm'])
    return lst
locs=sorted(os.listdir(base))
full={l:rows(l) for l in locs}
def parent(l):
    return l[:l.rfind('-')] if '-' in l else None
out={}
for l in locs:
    p=parent(l)
    # resolve parent through omission chain
    anc=p
    while anc is not None and anc not in full: anc=parent(anc)
    if anc is not None and full[anc]==full[l]:
        continue
    out[l]=full[l]
# compare 8 columns with the existing table
mism=[l for l in old if l in full and full[l][:8]!=old[l]]
print('old rows',len(old),'new rows',len(out),'8-col mismatches',len(mism),mism[:5], file=sys.stderr)
missing_old=[l for l in old if l not in out]; extra=[l for l in out if l not in old]
print('rows now kept that old omitted', len(extra), extra[:10], 'old rows omitted now', len(missing_old), missing_old[:10], file=sys.stderr)
with open(sys.argv[2],'w',encoding='utf-8') as fo:
    for l in sorted(out):
        fo.write(l+'\t'+'\t'.join(out[l])+'\n')
