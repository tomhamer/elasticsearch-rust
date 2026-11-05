#!/usr/bin/env python3
"""Materialize a gate plan from reviewed, concrete EC2 fixture bindings.
Does not launch commands, connect to hosts, compile, or run tests.
"""
import argparse, json, pathlib, re
from common import pin, save, sha
from run import REQUIRED

def artifact(path):
    p=pathlib.Path(path).resolve(strict=True)
    return {'path':str(p),'sha256':sha(p)}

def main():
    if not __debug__: raise RuntimeError('Plan preparation requires Python assertions; unset PYTHONOPTIMIZE and omit -O')
    parser=argparse.ArgumentParser();parser.add_argument('--bindings',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
    binding=json.loads(pathlib.Path(args.bindings).read_text());out=pathlib.Path(args.out);assert not out.exists()
    for key in ('candidate_revision','master_revision'):assert re.fullmatch('[0-9a-f]{40}',binding[key])
    stages=binding['stages'];assert {s['kind'] for s in stages}==REQUIRED and len(stages)==len(REQUIRED)
    assert stages[0]['kind']=='unit-integration' and stages[-1]['kind']=='profile'
    root=pathlib.Path(binding['source_root']).resolve(strict=True)
    # Full manifest, not just touched files. JSON maps source-relative names to
    # SHA256; the archive producer must exclude binary targets and secret files.
    manifest=pin(binding['source_manifest']);entries=json.loads(manifest.read_text());assert entries
    files=[]
    for name,digest in entries.items():
        path=(root/name).resolve(strict=True);assert path.is_relative_to(root)
        item={'path':str(path),'sha256':digest};pin(item);files.append(item)
    artifacts=[]
    for stage in stages:
        stage['source_revision']=binding['candidate_revision'];stage['master_revision']=binding['master_revision']
        assert re.fullmatch('[0-9a-f]{64}',stage['binary_sha256'])
        adapter=artifact(stage['adapter_path']);stage['adapter']=adapter;del stage['adapter_path'];artifacts.append(adapter)
        assert stage['argv'][0]==adapter['path']
        assert pathlib.Path(stage['report']).is_absolute() and not pathlib.Path(stage['report']).exists()
        stage['fixtures']=[artifact(path) for path in stage.pop('fixture_paths')]
    # Pin imports as well as entrypoints, and every native script/config listed
    # by the reviewed fixture author. No generated executable is inferred.
    artifacts += [artifact(path) for path in binding['artifact_paths']]
    artifacts += [artifact(path) for path in pathlib.Path(__file__).parent.glob('*.py')]
    plan={k:binding[k] for k in ('repository','candidate_revision','master_revision','hosts')}
    plan.update(stages=stages,source_files=files,source_manifest=binding['source_manifest'],source_manifest_sha256=sha(manifest),artifacts=artifacts)
    save(out,plan)
if __name__=='__main__':main()
