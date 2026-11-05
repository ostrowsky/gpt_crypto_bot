"""IC-1S official Spot archive diagnostic; never a native Demo fill source."""
import argparse
from collections import defaultdict
import csv
from datetime import date, datetime, timezone
import hashlib
import http.client
import io
import json
from pathlib import Path
import re
import sqlite3
import ssl
import time
from urllib.parse import quote, unquote
import uuid
import zipfile

from .adapter import HttpResponse
from .intrabar_probe import _seal, _read_sealed, _write, _bytes_sha, _single_writer, AMBIGUOUS
from .intrabar_data import _row, validate_seconds
from .intrabar_census import first_touch, census_counts
from .execution_order_study import verify_files
from .public_research_audit import descriptor_matches, PublicResearchJournal, verify_committed_probe_audit

HOST='data.binance.vision'
ZIP_LIMIT=16*1024*1024
CACHE_LIMIT=1024*1024*1024


def archive_path(symbol,day,*,checksum=False):
    if (not isinstance(symbol,str) or not symbol.isalnum() or not symbol.endswith('USDT') or len(symbol)>64 or
        not isinstance(day,str) or not re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}',day)):
        raise ValueError('ARCHIVE_ARGUMENTS')
    date.fromisoformat(day)
    raw=f'/data/spot/daily/klines/{symbol}/1s/{symbol}-1s-{day}.zip'+('.CHECKSUM' if checksum else '')
    return quote(raw,safe='/.-')


def _path(path):
    if not isinstance(path,str):raise ValueError('ARCHIVE_FORBIDDEN_PATH')
    decoded=unquote(path,errors='strict')
    match=re.fullmatch(r'/data/spot/daily/klines/([^/]+)/1s/[^/]+-1s-([0-9]{4}-[0-9]{2}-[0-9]{2})\.zip(\.CHECKSUM)?',decoded)
    if not match or archive_path(match[1],match[2],checksum=bool(match[3]))!=path:
        raise ValueError('ARCHIVE_FORBIDDEN_PATH')
    return match[1],match[2],bool(match[3])


def read_seconds_zip(body,checksum,symbol,day,origins):
    archive_path(symbol,day)
    if not isinstance(body,bytes) or len(body)>ZIP_LIMIT:raise ValueError('ARCHIVE_ZIP_SIZE')
    filename=f'{symbol}-1s-{day}.zip'
    try:
        match=re.fullmatch(r'([a-fA-F0-9]{64})[ \t]+\*?([^\r\n]+)',checksum.decode('utf-8').strip())
        if not match or match[2]!=filename or match[1].lower()!=hashlib.sha256(body).hexdigest():
            raise ValueError('ARCHIVE_CHECKSUM')
        at=int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp()*1000)
        scale=1000 if day>='2025-01-01' else 1
        selected={origin:[] for origin in origins}
        if any(type(o) is not int or not at<=o<at+86400000 or o%900000 for o in origins):
            raise ValueError('ARCHIVE_WINDOW_DATE')
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            members=archive.infolist()
            if (len(members)!=1 or members[0].filename!=filename[:-4]+'.csv' or members[0].is_dir() or
                members[0].flag_bits&1 or members[0].file_size>64*1024*1024):
                raise ValueError('ARCHIVE_MEMBER_SCHEMA')
            previous=None
            with archive.open(members[0]) as raw:
                for number,row in enumerate(csv.reader(io.TextIOWrapper(raw,encoding='utf-8',newline='')),1):
                    if number>86400 or len(row)!=12 or any(not re.fullmatch(r'[0-9]+',row[k]) for k in (0,6,8)):
                        raise ValueError('ARCHIVE_CSV_SCHEMA')
                    opening,closing=int(row[0]),int(row[6])
                    if opening%(1000*scale) or closing!=opening+1000*scale-1:
                        raise ValueError('ARCHIVE_TIMESTAMP_QUANTUM')
                    opening//=scale;closing//=scale
                    if not at<=opening<at+86400000 or previous is not None and opening<=previous:
                        raise ValueError('ARCHIVE_ORDER_OR_DAY')
                    normalized=[opening,*row[1:6],closing,row[7],int(row[8]),*row[9:]]
                    _row(normalized,1000,opening)
                    origin=opening//900000*900000
                    if origin in selected:selected[origin].append(normalized)
                    previous=opening
        return selected
    except (zipfile.BadZipFile,UnicodeError,csv.Error,RuntimeError,EOFError) as error:
        raise ValueError('ARCHIVE_UNREADABLE') from error


def refine_case(case,original,minutes,seconds,*,received_at_ms,verified=True):
    check=validate_seconds(original,minutes,seconds,received_at_ms=received_at_ms)
    ordering=first_touch(seconds,case['stop'],case['target'],resolution_ms=1000,
                         verified=verified and check['state']=='MATCH')
    return dict(source_check=check,ordering=ordering,source_kind='OFFICIAL_BINANCE_ARCHIVE_1s',
                actual_execution_reconstructed=False,promotion_allowed=False)


class ArchiveTransport:
    """Only the fixed official archive host; no redirects, proxies or auth."""
    def request(self,path,timeout_s):
        _path(path)
        if not 0<timeout_s<=15:raise ValueError('ARCHIVE_TIMEOUT')
        deadline=time.monotonic()+timeout_s
        def remaining():
            value=deadline-time.monotonic()
            if value<=0:raise TimeoutError()
            return value
        connection=http.client.HTTPSConnection(HOST,443,context=ssl.create_default_context(),timeout=min(5,remaining()))
        try:
            connection.connect();connection.sock.settimeout(remaining())
            connection.request('GET',path,headers={})
            connection.sock.settimeout(remaining());response=connection.getresponse()
            limit=4096 if path.endswith('.CHECKSUM') else ZIP_LIMIT
            chunks=[];size=0
            while True:
                sock=connection.sock or getattr(getattr(response.fp,'raw',None),'_sock',None)
                if sock is not None:sock.settimeout(remaining())
                block=response.read1(min(65536,limit+1-size))
                if not block:break
                size+=len(block)
                if size>limit:raise ValueError('ARCHIVE_BODY_LIMIT')
                chunks.append(block)
            remaining()
            return HttpResponse(response.status,b''.join(chunks),dict(response.getheaders()))
        finally:connection.close()


class ArchiveClient:
    def __init__(self,folder,audit,*,transport=None,monotonic=time.monotonic,sleep=time.sleep,max_attempts=2000,max_wall_s=3600):
        from .interaction_audit import AuditJournal
        if (not isinstance(audit,AuditJournal) or type(max_attempts) is not int or not 1<=max_attempts<=2000 or
            not 0<max_wall_s<=3600):raise ValueError('ARCHIVE_INVOCATION')
        self.folder=Path(folder);self.audit=audit;self.transport=transport or ArchiveTransport()
        self.monotonic,self.sleep=monotonic,sleep;self.deadline=monotonic()+max_wall_s
        self.max_attempts=max_attempts;self.attempts=0;self.last=None;self.blocked=None;self.last_reason=None
        self.bytes=sum(p.stat().st_size for p in (self.folder/'files').glob('*.bin'))

    def fetch(self,path):
        _path(path);key=hashlib.sha256(path.encode()).hexdigest();cache=self.folder/'responses'/(key+'.json')
        if cache.exists():
            value=_read_sealed(cache);_public_file(self.folder,value)
            if value['path']!=path:raise ValueError('ARCHIVE_CACHE_PATH')
            return value
        request_id=uuid.uuid4().hex
        for number in range(1,4):
            if self.blocked or self.attempts>=self.max_attempts or self.monotonic()>=self.deadline:
                self.last_reason=self.blocked or 'ARCHIVE_INVOCATION_BUDGET';return None
            delay=max(0.,.3-(self.monotonic()-self.last)) if self.last is not None else 0.
            if delay:self.sleep(delay)
            remaining=self.deadline-self.monotonic()
            if remaining<=0:self.last_reason='ARCHIVE_INVOCATION_BUDGET';return None
            attempt=uuid.uuid4().hex
            begin=self.audit.append('REST_BEGIN',dict(method='GET',endpoint='https://'+HOST+path,private=False,
                request_id=request_id,attempt_id=attempt,parameters=dict(archive_path=path),attempt_number=number))
            self.attempts+=1;self.last=self.monotonic();response=None;error=None;payload=None
            try:
                response=self.transport.request(path,min(15,remaining))
                limit=4096 if path.endswith('.CHECKSUM') else ZIP_LIMIT
                if not isinstance(response.body,bytes) or len(response.body)>limit:raise ValueError('ARCHIVE_BODY_LIMIT')
                digest=hashlib.sha256(response.body).hexdigest();raw=self.folder/'files'/(digest+'.bin')
                if not raw.exists():
                    if self.bytes+len(response.body)>CACHE_LIMIT:
                        self.blocked='ARCHIVE_DISK_BUDGET';raise ValueError('ARCHIVE_DISK_BUDGET')
                    raw.parent.mkdir(parents=True,exist_ok=True);temp=raw.with_suffix('.next');temp.write_bytes(response.body);temp.replace(raw)
                    self.bytes+=len(response.body)
                elif _bytes_sha(raw)!=digest:raise ValueError('ARCHIVE_EXISTING_FILE_CHANGED')
                payload=dict(representation='PUBLIC_FILE_REFERENCE',raw_file=str(raw.relative_to(self.folder)),
                             sha256=digest,bytes_n=len(response.body))
            except (OSError,TimeoutError,ValueError,http.client.HTTPException) as exc:
                error=type(exc).__name__
            received=time.time_ns()//1000000
            end=self.audit.append('REST_END',dict(method='GET',endpoint='https://'+HOST+path,private=False,
                request_id=request_id,attempt_id=attempt,parent_event_id=begin,
                outcome='RESPONSE' if error is None else 'TRANSPORT_ERROR',status=response.status if response is not None else None,
                payload=payload,received_at_ms=received,error_kind=error))
            receipt=dict(path=path,url='https://'+HOST+path,request_id=request_id,attempt_id=attempt,
                begin_event_id=begin,end_event_id=end,status=response.status if response is not None else None,
                received_at_ms=received,error_kind=error,**(payload or {}))
            _seal(self.folder/'requests'/(attempt+'.json'),receipt)
            if error is None and response.status==200:
                _seal(cache,receipt);return receipt
            self.last_reason=error or f'HTTP_{response.status}'
            if response is not None and response.status==418:self.blocked='ARCHIVE_IP_BAN';return None
            delay=min(4.,2.**(number-1))
            if response is not None and response.status==429:
                headers={k.lower():v for k,v in response.headers.items()}
                try:delay=max(delay,float(headers.get('retry-after','1')))
                except ValueError:delay=61
                if not 0<=delay<=60:self.blocked='ARCHIVE_LONG_COOLDOWN';return None
            elif error is None and response.status<500:return None
            if number<3:
                if delay>=self.deadline-self.monotonic():self.blocked='ARCHIVE_DEADLINE_COOLDOWN';return None
                self.sleep(delay)
        return None


def _public_file(folder,receipt):
    path=(Path(folder)/receipt['raw_file']).resolve()
    if not path.is_relative_to((Path(folder)/'files').resolve()) or not path.is_file():raise ValueError('ARCHIVE_FILE_SCOPE')
    if path.stat().st_size!=receipt['bytes_n'] or _bytes_sha(path)!=receipt['sha256']:raise ValueError('ARCHIVE_FILE_HASH')
    return path


def verify_archive_audit(folder,path):
    folder,path=Path(folder),Path(path);failures=[];attempts=errors=linked=0
    try:
        if path.with_name(path.name+'-wal').exists() and path.with_name(path.name+'-wal').stat().st_size:
            return dict(state='UNKNOWN',reasons=['ARCHIVE_AUDIT_WAL_OPEN'])
        connection=sqlite3.connect(path.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
        tip='0'*64;begins={};ends={};runs={};records={}
        try:
            for seq,identifier,kind,payload,sha in connection.execute('SELECT seq,id,kind,payload_json,content_sha256 FROM records ORDER BY seq'):
                record=json.loads(payload)
                canonical=json.dumps(record,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
                if (seq!=len(records)+1 or record['seq']!=seq or record['event_id']!=identifier or record['kind']!=kind or
                    record['previous_sha256']!=tip or hashlib.sha256(canonical).hexdigest()!=sha or
                    record.get('application_id')!='binance_demo_bot' or record.get('environment')!='BINANCE_SPOT_DEMO'):
                    raise ValueError('ARCHIVE_AUDIT_CHAIN')
                tip=sha;records[identifier]=record;data=record['data'];run=record['run_id']
                if kind=='RUN_BEGIN':
                    if run in runs:raise ValueError('ARCHIVE_DUPLICATE_RUN')
                    runs[run]=False;continue
                if run not in runs or runs[run]:raise ValueError('ARCHIVE_RUN_BOUNDARY')
                if kind=='RUN_END':runs[run]=True;continue
                endpoint=data.get('endpoint','')
                if (kind not in ('REST_BEGIN','REST_END') or data.get('method')!='GET' or data.get('private') is not False or
                    not re.fullmatch('[a-f0-9]{64}',data.get('public_descriptor_sha256','')) or
                    endpoint!='[REDACTED]' and not endpoint.startswith('https://'+HOST+'/')):raise ValueError('ARCHIVE_AUDIT_ENDPOINT')
                if endpoint!='[REDACTED]':_path(endpoint[len('https://'+HOST):])
                if kind=='REST_BEGIN':begins[identifier]=record;attempts+=1
                else:
                    begin=begins.get(data.get('parent_event_id'))
                    if (begin is None or begin['run_id']!=run or data['parent_event_id'] in ends or
                        any(data[k]!=begin['data'][k] for k in ('request_id','attempt_id','endpoint'))):
                        raise ValueError('ARCHIVE_ATTEMPT_PAIR')
                    ends[data['parent_event_id']]=record
                    errors+=int(data.get('outcome')!='RESPONSE' or data.get('status')!=200)
        finally:connection.close()
        if not runs or not all(runs.values()) or set(begins)!=set(ends):raise ValueError('ARCHIVE_AUDIT_INCOMPLETE')
        requests=list((folder/'requests').glob('*.json'))
        if len(requests)!=len(ends):raise ValueError('ARCHIVE_RECEIPT_COUNT')
        for p in requests:
            receipt=_read_sealed(p);begin=begins.get(receipt['begin_event_id']);end=records.get(receipt['end_event_id'])
            if (begin is None or end is None or ends.get(begin['event_id'])!=end or
                any(receipt[k]!=end['data'][k] for k in ('request_id','attempt_id','status','received_at_ms','error_kind')) or
                receipt['url']!='https://'+HOST+receipt['path'] or
                not descriptor_matches(begin['data'],receipt['url'],{'archive_path':receipt['path']}) or
                not descriptor_matches(end['data'],receipt['url'])):raise ValueError('ARCHIVE_RECEIPT_BINDING')
            _path(receipt['path'])
            payload=end['data'].get('payload')
            if payload is not None:
                if payload.get('representation')!='PUBLIC_FILE_REFERENCE' or any(receipt[k]!=payload[k] for k in ('raw_file','sha256','bytes_n')):
                    raise ValueError('ARCHIVE_BINARY_BINDING')
                _public_file(folder,receipt)
        for p in (folder/'responses').glob('*.json'):
            cached=_read_sealed(p)
            if cached['status']!=200 or cached['error_kind'] is not None or _read_sealed(folder/'requests'/(cached['attempt_id']+'.json'))!=cached:
                raise ValueError('ARCHIVE_CACHE_BINDING')
            _public_file(folder,cached);linked+=1
    except (OSError,sqlite3.Error,ValueError,TypeError,KeyError) as error:failures.append(str(error) if isinstance(error,ValueError) else type(error).__name__)
    return dict(state='FAIL' if failures else 'VERIFIED_PUBLIC_ARCHIVE_ATTEMPTS',reasons=failures,
        attempts_n=attempts,errors_n=errors,linked_files_n=linked,scope='PUBLIC_ARCHIVE_FILES_AND_REQUESTS_ONLY',promotion_allowed=False)


@_single_writer
def execute_archive(folder,native_folder,truth_report):
    folder=Path(folder);native=Path(native_folder).resolve();truth_report=Path(truth_report).resolve()
    app=Path(__file__).resolve().parents[2]
    def status(phase,**extra):
        value=dict(phase=phase,at_ms=time.time_ns()//1000000,promotion_allowed=False,**extra)
        _write(folder/'status.json',value);print(json.dumps(value),flush=True)
    try:
        status('REGISTERING_ARCHIVE_SOURCE')
        parent=_read_sealed(native/'registration.json');census=_read_sealed(native/'census.json')
        if verify_committed_probe_audit(native,native/'rest-audit.sqlite')['state']!='VERIFIED_RECORDED_PUBLIC_ATTEMPTS':
            raise ValueError('ARCHIVE_NATIVE_PARENT_UNVERIFIED')
        required={(c['symbol'],c['origin_ms']) for c in census['cases'] if c['source_state']=='MATCH' and
                  c['ordering']['state'].startswith('ORDER_UNRESOLVED')}
        required.update(AMBIGUOUS)
        originals={(w['symbol'],w['origin_ms']):w for w in parent['windows']}
        accepted={(c['symbol'],c['origin_ms']):c for c in census['cases'] if c['source_state']=='MATCH'}
        files={str(native/name):_bytes_sha(native/name) for name in ('registration.json','census.json','audit-check.json','source-check.json','rest-audit.sqlite')}
        windows=[];days=defaultdict(set)
        for key in sorted(required):
            if key not in accepted:raise ValueError('ARCHIVE_REQUIRED_MINUTE_MISSING')
            case=accepted[key];response=Path(case['response_file']);files[str(response)]=_bytes_sha(response)
            files[str(response.with_suffix('.json.sha256'))]=_bytes_sha(response.with_suffix('.json.sha256'))
            day=datetime.fromtimestamp(key[1]/1000,timezone.utc).date().isoformat();days[key[0],day].add(key[1])
            windows.append(dict(originals[key],minute_response_file=str(response),day=day))
        for name in ('archive_seconds.py','intrabar_census.py','intrabar_data.py','intrabar_probe.py','intrabar_audit.py','public_research_audit.py','interaction_audit.py'):
            p=Path(__file__).with_name(name);files[str(p)]=_bytes_sha(p)
        spec=Path(parent['spec_path']);files[str(spec)]=_bytes_sha(spec);files[str(truth_report)]=_bytes_sha(truth_report)
        audit_spec=spec.with_name('binance-demo-public-request-commitments.md');files[str(audit_spec)]=_bytes_sha(audit_spec)
        truth=json.loads(truth_report.read_text(encoding='utf-8'))
        if truth['profile']!='full':raise ValueError('ARCHIVE_FULL_TRUTH_REQUIRED')
        binding=dict(protocol='IC-1-ARCHIVE-SECONDS-20261105',source='OFFICIAL_BINANCE_SPOT_ARCHIVE_1s',
            native_folder=str(native),files=files,windows=windows,
            downloads=[dict(symbol=s,day=d,origins=sorted(ats)) for (s,d),ats in sorted(days.items())],
            full_truth_harness=truth['status'].upper(),actual_execution_reconstructed=False,promotion_allowed=False)
        _seal(folder/'registration.json',binding);verify_files(files)
        from .isolation import AppPaths
        audit=PublicResearchJournal(AppPaths(app),account_alias='public-research',episode_id=folder.name,
            scope_id='official-archive-seconds',database=str((folder/'audit.sqlite').relative_to(app/'.runtime')),max_bytes=128*1024*1024)
        client=ArchiveClient(folder,audit);receipts={}
        try:
            for i,item in enumerate(binding['downloads']):
                key=(item['symbol'],item['day']);pair=[]
                for checksum in (True,False):
                    value=client.fetch(archive_path(*key,checksum=checksum));pair.append(value)
                    if value is None:break
                receipts[key]=pair
                if i%10==0:status('COLLECTING_ARCHIVES',files_done_n=i+1,files_N=len(binding['downloads']),native_attempts_n=client.attempts)
        finally:audit.close()
        certification=verify_archive_audit(folder,folder/'audit.sqlite');_write(folder/'audit-check.json',certification)
        verified=certification['state']=='VERIFIED_PUBLIC_ARCHIVE_ATTEMPTS';results={};checks=[]
        for i,item in enumerate(binding['downloads']):
            pair=receipts[item['symbol'],item['day']];selected=None;reason=None
            if verified and len(pair)==2 and all(pair):
                try:
                    selected=read_seconds_zip(_public_file(folder,pair[1]).read_bytes(),_public_file(folder,pair[0]).read_bytes(),
                        item['symbol'],item['day'],item['origins'])
                except (ValueError,ArithmeticError) as error:reason=type(error).__name__+':'+str(error)
            else:reason='ARCHIVE_UNAVAILABLE_OR_UNVERIFIED'
            for window in (w for w in windows if w['symbol']==item['symbol'] and w['day']==item['day']):
                key=window['symbol'],window['origin_ms']
                minutes=_read_sealed(window['minute_response_file'])['rows']
                seconds=selected[key[1]] if selected is not None else []
                check=validate_seconds(window['row'],minutes,seconds,received_at_ms=pair[-1]['received_at_ms'] if pair and pair[-1] else 0)
                checks.append(dict(symbol=key[0],origin_ms=key[1],source_check=check,download_error=reason))
                if check['state']=='MATCH' and verified:
                    path=folder/'windows'/(hashlib.sha256(f'{key[0]}:{key[1]}'.encode()).hexdigest()+'.json')
                    _seal(path,dict(symbol=key[0],origin_ms=key[1],rows=seconds,source_check=check))
                    results[key]=str(path)
            if i%10==0:status('VALIDATING_ARCHIVES',files_done_n=i+1,files_N=len(binding['downloads']))
        merged=[];newly_resolved=0
        for case in census['cases']:
            row=dict(case);key=case['symbol'],case['origin_ms']
            if case['ordering']['state'].startswith('ORDER_UNRESOLVED') and key in results:
                row['ordering']=first_touch(_read_sealed(results[key])['rows'],case['stop'],case['target'],resolution_ms=1000)
                row['source_kind']='OFFICIAL_BINANCE_ARCHIVE_1s'
                newly_resolved+=int(row['ordering']['state'] in ('SL_FIRST','TP_FIRST'))
            merged.append(row)
        verify_files(files)
        groups=defaultdict(list)
        for c in merged:groups[c['segment'],c['cost'],c['order'],c['label']].append(c)
        grouped=[dict(segment=k[0],cost=k[1],order=k[2],label=k[3],**census_counts(v)) for k,v in sorted(groups.items())]
        report=dict(protocol=binding['protocol'],**census_counts(merged),minute_resolved_n=census['resolved_n'],
            archive_newly_resolved_n=newly_resolved,archive_requested_windows_n=len(windows),archive_matched_windows_n=len(results),
            audit_state=certification['state'],source_checks=checks,groups=grouped,cases=merged,
            evidence_kind='SIMULATION_DATA_DIAGNOSTIC',full_truth_harness=binding['full_truth_harness'],
            actual_execution_reconstructed=False,full_replay_complete=False,promotion_allowed=False)
        _seal(folder/'report.json',report)
        lines=['# IC-1: проверка первого касания уровней','',
            f"Все {census['windows_n']} исходных окон / {report['cases_n']} зависимых привязок. По Demo1m разрешено {census['resolved_n']}; по отдельно проверенному архиву1s дополнительно {newly_resolved}; осталось {report['unresolved_n']}.",
            f"Архивных окон MATCH {len(results)}/{len(windows)}. Provenance {certification['state']}. Полный Truth Harness **{binding['full_truth_harness']}**.",
            '', '| Период | Издержки | Допущение | Метод | N | Стоп раньше | Цель раньше | Неясно | Противоречий |',
            '|---|---|---|---|---:|---:|---:|---:|---:|']
        for g in grouped:lines.append(f"| {g['segment']} | {g['cost']} | {g['order']} | {g['label']} | {g['cases_n']} | {g['sl_first_n']} | {g['tp_first_n']} | {g['unresolved_n']} | {g['contradicted_n']} |")
        lines+=['', 'Источник1s — отдельный официальный Spot-архив, с контрольной суммой ZIP и проверкой каждой секунды против исходной15m и всех пятнадцати Demo1m.',
            'Совпадение относится только к проверенным окнам и не делает архив нативной историей Demo-исполнений.',
            'Оба уровня в одной секунде остаются неизвестными. Ошибки/отсутствующие файлы не превращены в плоский путь.',
            'PnL не пересчитан: изменённые выходы требуют полного причинного портфельного replay. Модели, пороги, риск и торговля этим экспериментом не изменены.',
            'Нужен отдельный полный replay с замороженной политикой и явно зафиксированной обработкой оставшихся неопределённостей. Прибыль и ежедневный доход не установлены.']
        (folder/'findings-ru.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        status('COMPLETE_DATA_DIAGNOSTIC',resolved_n=report['resolved_n'],unresolved_n=report['unresolved_n'],archive_matches_n=len(results))
    except BaseException as error:status('INCOMPLETE',error_kind=type(error).__name__);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',required=True);parser.add_argument('--native-folder',required=True);parser.add_argument('--truth-report',required=True)
    args=parser.parse_args();folder=Path(args.folder).resolve();app=Path(__file__).resolve().parents[2]
    if not folder.is_relative_to(app/'.runtime/intrabar-execution') or not folder.name.startswith('IC-1-20261105'):
        raise SystemExit('OWN_ARCHIVE_RUNTIME_REQUIRED')
    execute_archive(folder,args.native_folder,args.truth_report)


if __name__=='__main__':main()
