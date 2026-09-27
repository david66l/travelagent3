"""Version an internal request-count limit without changing money or history.

Run on the existing cloud host with no paid queue in flight. An interrupted
migration can only roll forward using its recorded old/new configurations.
"""
import argparse,fcntl,hashlib,json,os
from pathlib import Path


def atomic(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as file:
        json.dump(value,file,ensure_ascii=False,indent=2);file.flush();os.fsync(file.fileno())
    temp.replace(path)


def migrate(ledger_path, config_path, record_path, new_limit):
    with ledger_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger=json.loads(ledger_path.read_text());config=json.loads(config_path.read_text())
        if any(v['status']=='reserved' for v in ledger['requests'].values()):raise ValueError('Requests still in flight')
        if ledger.get('frozen'):raise ValueError('Frozen billing incident requires separate investigation')
        if record_path.exists():
            record=json.loads(record_path.read_text())
            if record['new_config']['max_requests']!=new_limit:raise ValueError('Migration target changed')
            old,new=record['old_config'],record['new_config']
            if ledger['config'] not in [old,new] or config not in [old,new]:raise ValueError('Unrelated configuration drift')
        else:
            if ledger['config']!=config:raise ValueError('Configuration already inconsistent')
            if config['limit_micros']!=100_000_000 or type(new_limit) is not int or not config['max_requests']<new_limit<=15000:
                raise ValueError('Only a bounded count increase under the existing 100 CNY cap is supported')
            old=config;new={**old,'max_requests':new_limit}
            record=dict(schema_version='api-request-count-migration.v1',old_config=old,new_config=new,
                request_history_sha256=hashlib.sha256(json.dumps(ledger['requests'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                original_request_count=len(ledger['requests']),original_charged_micros=sum(v['charged_micros'] for v in ledger['requests'].values()),completed=False)
            atomic(record_path,record)
        digest=hashlib.sha256(json.dumps(ledger['requests'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if digest!=record['request_history_sha256']:raise ValueError('Request history changed during migration')
        ledger['config']=new;atomic(ledger_path,ledger);atomic(config_path,new)
        record['completed']=True;atomic(record_path,record)
        return record


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['ledger','config','record']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--max-requests',type=int,required=True)
    args=parser.parse_args();result=migrate(args.ledger,args.config,args.record,args.max_requests)
    print(json.dumps({k:result[k] for k in ['completed','original_request_count','original_charged_micros']}))
