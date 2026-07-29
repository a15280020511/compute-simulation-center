#!/usr/bin/env python3
"""Allowlisted OpenAlex + Crossref metadata evidence builder."""
from __future__ import annotations
import argparse, hashlib, json, urllib.parse, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ALLOWED_HOSTS={'api.openalex.org','api.crossref.org'}
class LiteratureEvidenceError(ValueError): pass

def _get(url: str, timeout: int = 20) -> dict[str, Any]:
    parsed=urllib.parse.urlparse(url)
    if parsed.scheme!='https' or parsed.hostname not in ALLOWED_HOSTS: raise LiteratureEvidenceError('host is not allowlisted')
    request=urllib.request.Request(url,headers={'User-Agent':'decision-system-literature-evidence/1.0 (mailto:repository-owner@example.invalid)'})
    with urllib.request.urlopen(request,timeout=timeout) as response:
        if response.status!=200: raise LiteratureEvidenceError(f'HTTP {response.status}')
        data=response.read(5_000_001)
    if len(data)>5_000_000: raise LiteratureEvidenceError('response exceeds 5 MB')
    value=json.loads(data); return value

def build(query: str, per_page: int = 20) -> dict[str, Any]:
    query=query.strip()
    if not query or len(query)>500 or per_page<1 or per_page>50: raise LiteratureEvidenceError('invalid query or per_page')
    openalex_url='https://api.openalex.org/works?'+urllib.parse.urlencode({'search':query,'per_page':per_page,'sort':'cited_by_count:desc'})
    oa=_get(openalex_url); results=[]; seen=set()
    for work in oa.get('results',[]):
        doi=(work.get('doi') or '').replace('https://doi.org/','').lower()
        if not doi or doi in seen: continue
        seen.add(doi); cr=_get('https://api.crossref.org/works/'+urllib.parse.quote(doi,safe=''))
        message=cr.get('message') or {}; updates=message.get('update-to') or []
        retracted=any(str(item.get('type','')).lower()=='retraction' or str(item.get('source','')).lower()=='retraction-watch' for item in updates if isinstance(item,dict))
        results.append({'doi':doi,'title':work.get('title'),'publication_year':work.get('publication_year'),'cited_by_count':work.get('cited_by_count'),'openalex_id':work.get('id'),'crossref_type':message.get('type'),'publisher':message.get('publisher'),'license':message.get('license'),'updates':updates,'retracted_or_retraction_update':retracted,'parameter_status':'literature-raw-result-only'})
    payload={'schema_version':'literature-evidence-package-v1','created_at':datetime.now(timezone.utc).isoformat(),'query':query,'sources':['OpenAlex','Crossref'],'records':results,'automatic_parameter_promotion_allowed':False}
    payload['content_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest(); return payload

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--query',required=True); p.add_argument('--output',required=True); p.add_argument('--per-page',type=int,default=20); a=p.parse_args()
    Path(a.output).write_text(json.dumps(build(a.query,a.per_page),ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); return 0
if __name__=='__main__': raise SystemExit(main())
