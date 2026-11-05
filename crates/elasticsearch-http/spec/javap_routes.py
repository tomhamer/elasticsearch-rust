#!/usr/bin/env python3
"""javap every class of an ES distribution that references `RestHandler$Route`
(the classes declaring REST routes), for routes_from_javap.py.

usage: javap_routes.py <ES home> <work dir> <out javap.txt>
  e.g. javap_routes.py ~/ref/elasticsearch-9.5.5 ~/routes9/cls ~/routes9/javap.txt
"""
import glob, os, subprocess, sys, zipfile

home, work, out = (os.path.expanduser(a) for a in sys.argv[1:4])
jars = glob.glob(home + '/lib/**/*.jar', recursive=True) + glob.glob(home + '/modules/**/*.jar', recursive=True)
classes = []
for j in jars:
    with zipfile.ZipFile(j) as z:
        for e in z.namelist():
            if e.endswith('.class') and e.startswith('org/elasticsearch'):
                b = z.read(e)
                if b'org/elasticsearch/rest/RestHandler$Route' in b:
                    p = os.path.join(work, e)
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    open(p, 'wb').write(b)
                    classes.append(e)
classes.sort()
with open(out, 'w') as f:
    subprocess.run([home + '/jdk/bin/javap', '-c', '-p', '-constants'] + classes, cwd=work, stdout=f, stderr=subprocess.STDOUT, check=False)
print(len(jars), 'jars', len(classes), 'classes ->', out)
