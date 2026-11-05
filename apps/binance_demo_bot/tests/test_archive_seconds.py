"""IC07..10: independently-labelled public archive and exact temporal schema."""
import csv
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

import _support
from binance_demo_bot.adapter import HttpResponse
from binance_demo_bot.interaction_audit import AuditJournal
from binance_demo_bot.isolation import AppPaths
from binance_demo_bot.public_research_audit import PublicResearchJournal
from binance_demo_bot.archive_seconds import (archive_path, read_seconds_zip,
    ArchiveClient, verify_archive_audit, refine_case)


def row(at,us=True):
    scale=1000 if us else 1
    return [str(at*scale),'100','100','100','100','1',str((at+1000)*scale-1),'100','1','0','0','0']


def zipped(rows,symbol='AAAUSDT',day='2025-01-01',extra=None):
    name=f'{symbol}-1s-{day}.csv'; s=io.StringIO();csv.writer(s,lineterminator='\n').writerows(rows)
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr(name,s.getvalue())
        if extra:z.writestr(extra,'bad')
    raw=b.getvalue();check=f'{hashlib.sha256(raw).hexdigest()}  {name[:-4]}.zip\n'.encode()
    return raw,check


class Clock:
    def __init__(self):self.now=0.;self.sleeps=[]
    def mono(self):return self.now
    def sleep(self,value):self.sleeps.append(value);self.now+=value


class Transport:
    def __init__(self,responses):self.responses=list(responses);self.calls=[]
    def request(self,path,timeout_s):
        self.calls.append((path,timeout_s));response=self.responses.pop(0)
        if isinstance(response,BaseException):raise response
        return response


class ArchiveDataTests(unittest.TestCase):
    def test_IC08_fixed_path_and_unicode(self):
        self.assertEqual(archive_path('AAAUSDT','2025-01-01'),'/data/spot/daily/klines/AAAUSDT/1s/AAAUSDT-1s-2025-01-01.zip')
        self.assertIn('%E7%89%9B',archive_path('牛USDT','2025-01-01'))
        self.assertTrue(archive_path('AAAUSDT','2025-01-01',checksum=True).endswith('.zip.CHECKSUM'))

    def test_IC08_injection_dates_and_symbols_reject_before_get(self):
        for symbol,day in (('../AAA','2025-01-01'),('AAA?x=1','2025-01-01'),('AAAUSDT','2025-02-30'),('AAAUSDT','2025-1-1')):
            with self.subTest(symbol=symbol,day=day),self.assertRaises(ValueError):archive_path(symbol,day)

    def test_IC09_microseconds_exact_normalization(self):
        at=1735689600000;raw,check=zipped([row(at),row(at+1000)])
        data=read_seconds_zip(raw,check,'AAAUSDT','2025-01-01',[at])
        self.assertEqual([r[0] for r in data[at]],[at,at+1000]);self.assertEqual(data[at][0][6],at+999)

    def test_IC09_pre2025_milliseconds(self):
        at=1704067200000;raw,check=zipped([row(at,False)],day='2024-01-01')
        self.assertEqual(read_seconds_zip(raw,check,'AAAUSDT','2024-01-01',[at])[at][0][0],at)

    def test_IC09_unicode_checksum_filename(self):
        at=1735689600000;raw,check=zipped([row(at)],symbol='牛USDT')
        self.assertEqual(read_seconds_zip(raw,check,'牛USDT','2025-01-01',[at])[at][0][0],at)

    def test_IC09_wrong_unit_or_fractional_open_rejects(self):
        at=1735689600000
        for rows in ([row(at,False)],[row(at+1)]):
            raw,check=zipped(rows)
            with self.subTest(rows=rows),self.assertRaises(ValueError):read_seconds_zip(raw,check,'AAAUSDT','2025-01-01',[at])

    def test_IC09_checksum_and_filename_are_both_required(self):
        at=1735689600000;raw,check=zipped([row(at)])
        for bad in (check.replace(b'AAAUSDT',b'BBBUSDT'),b'0'*64+b'  AAAUSDT-1s-2025-01-01.zip\n'):
            with self.assertRaises(ValueError):read_seconds_zip(raw,bad,'AAAUSDT','2025-01-01',[at])

    def test_IC09_duplicate_offday_extra_member_and_bad_schema_reject(self):
        at=1735689600000
        for rows,extra in (([row(at),row(at)],None),([row(at-1000)],None),([row(at)],'../evil.csv'),([row(at)[:-1]],None)):
            raw,check=zipped(rows,extra=extra)
            with self.subTest(rows=rows,extra=extra),self.assertRaises(ValueError):read_seconds_zip(raw,check,'AAAUSDT','2025-01-01',[at])

    def test_IC09_untrusted_header_not_guessed(self):
        raw,check=zipped([['open_time']*12])
        with self.assertRaises(ValueError):read_seconds_zip(raw,check,'AAAUSDT','2025-01-01',[1735689600000])

    def test_IC10_missing_or_mismatched_nested_data_cannot_resolve(self):
        case=dict(stop=99.5,target=100.5)
        original=[0,'100','101','99','100','900',899999,'90000',900,'0','0','0']
        minutes=[[i*60000,'100','101' if i==0 else '100','99' if i==0 else '100','100','60',i*60000+59999,'6000',60,'0','0','0'] for i in range(15)]
        seconds=[[i*1000,'100','101' if i==0 else '100','99' if i==0 else '100','100','1',i*1000+999,'100',1,'0','0','0'] for i in range(900)]
        good=refine_case(case,original,minutes,seconds,received_at_ms=1000000)
        self.assertEqual(good['source_check']['state'],'MATCH')
        self.assertEqual(good['ordering']['state'],'ORDER_UNRESOLVED_SAME_BUCKET')
        seconds[60][2]='102'
        bad=refine_case(case,original,minutes,seconds,received_at_ms=1000000)
        self.assertEqual(bad['ordering']['state'],'ORDER_UNRESOLVED_DATA')
        self.assertNotEqual(bad['source_check']['state'],'MATCH')


class ArchiveClientTests(unittest.TestCase):
    def setUp(self):
        self.app=Path(__file__).resolve().parents[1]
        self.tmp=tempfile.TemporaryDirectory(dir=self.app/'.runtime');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.clock=Clock()
        self.audit=PublicResearchJournal(AppPaths(self.app),account_alias='public-research',episode_id=self.root.name,
            scope_id='archive-test',database=str((self.root/'audit.sqlite').relative_to(self.app/'.runtime')))
        self.addCleanup(self.audit.close)

    def client(self,responses,**kwargs):
        transport=Transport(responses)
        client=ArchiveClient(self.root,self.audit,transport=transport,monotonic=self.clock.mono,sleep=self.clock.sleep,**kwargs)
        return client,transport

    def test_IC08_public_file_audit_cache_and_tamper(self):
        client,transport=self.client([HttpResponse(200,b'public-zip',{})])
        path=archive_path('AAAUSDT','2025-01-01');a=client.fetch(path);b=client.fetch(path)
        self.assertEqual(a,b);self.assertEqual(len(transport.calls),1)
        self.audit.close();check=verify_archive_audit(self.root,self.root/'audit.sqlite')
        self.assertEqual(check['state'],'VERIFIED_PUBLIC_ARCHIVE_ATTEMPTS')
        (self.root/a['raw_file']).write_bytes(b'changed')
        self.assertEqual(verify_archive_audit(self.root,self.root/'audit.sqlite')['state'],'FAIL')

    def test_IC08_no_redirect_and_no_credential_endpoint(self):
        client,transport=self.client([HttpResponse(302,b'redirect',{'Location':'https://evil.invalid'})])
        self.assertIsNone(client.fetch(archive_path('AAAUSDT','2025-01-01')))
        self.assertEqual(len(transport.calls),1)

    def test_PR03_masked_archive_url_requires_exact_commitment(self):
        client,transport=self.client([HttpResponse(200,b'public',{})])
        self.assertIsNotNone(client.fetch(archive_path('COOKIEUSDT','2025-01-01')))
        self.audit.close()
        self.assertEqual(verify_archive_audit(self.root,self.root/'audit.sqlite')['state'],'VERIFIED_PUBLIC_ARCHIVE_ATTEMPTS')
        for path in ('https://evil.invalid/a','/api/v3/account',archive_path('AAAUSDT','2025-01-01')+'?x=1'):
            with self.assertRaises(ValueError):client.fetch(path)
        self.assertEqual(len(transport.calls),1)

    def test_IC08_ban_long_cooldown_and_budget_latch(self):
        for status,headers in ((418,{}),(429,{'Retry-After':'120'})):
            client,transport=self.client([HttpResponse(status,b'no',headers)])
            self.assertIsNone(client.fetch(archive_path('AAAUSDT','2025-01-01')))
            self.assertIsNone(client.fetch(archive_path('BBBUSDT','2025-01-01')))
            self.assertEqual(len(transport.calls),1)

    def test_IC08_retry_is_bounded_paced_and_preserves_error(self):
        client,transport=self.client([TimeoutError(),HttpResponse(200,b'ok',{})],max_attempts=2)
        self.assertIsNotNone(client.fetch(archive_path('AAAUSDT','2025-01-01')))
        self.assertEqual(len(transport.calls),2);self.assertTrue(self.clock.sleeps)
        self.assertIsNone(client.fetch(archive_path('BBBUSDT','2025-01-01')))
        self.assertEqual(len(transport.calls),2)
        self.audit.close();check=verify_archive_audit(self.root,self.root/'audit.sqlite')
        self.assertEqual(check['state'],'VERIFIED_PUBLIC_ARCHIVE_ATTEMPTS');self.assertEqual(check['errors_n'],1)


if __name__=='__main__':unittest.main()
