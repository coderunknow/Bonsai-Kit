# Paste this entire file into ONE Colab/Kaggle Python cell.
import os, sys, subprocess, pathlib, json, time, socket, secrets, hashlib, re, platform, urllib.request, shutil, ctypes, importlib.util
from urllib.error import HTTPError

ROOT = pathlib.Path('/kaggle/working/bonsai2-api' if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') else '/content/bonsai2-api')
ROOT.mkdir(parents=True, exist_ok=True)
DEMO = ROOT / 'Bonsai-demo'
ALIAS = 'ternary-bonsai-2-27b'
REPO = 'prism-ml/Ternary-Bonsai-2-27B-gguf'
FILE = 'Ternary-Bonsai-2-27B-PQ2_0.gguf'
STATE = ROOT / 'state.json'

def phase(i, text): print(f'\n[{i}/8] {text}', flush=True)
def run(cmd, **kw): return subprocess.run(cmd, check=True, text=True, **kw)
def install(module, package):
    if importlib.util.find_spec(module) is None: run([sys.executable, '-m', 'pip', 'install', '-q', package])
def get(url, key=None, data=None, timeout=30):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
        headers={'Content-Type':'application/json', **({'Authorization':'Bearer '+key} if key else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return r.status, json.loads(r.read())
    except HTTPError as e: return e.code, e.read().decode(errors='replace')
def ensure(condition, message):
    if not condition: raise RuntimeError(message)
def free_port():
    s=socket.socket(); s.bind(('127.0.0.1',0)); p=s.getsockname()[1]; s.close(); return p

def memory():
    raw=run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version,pci.bus_id', '--format=csv,noheader,nounits'], capture_output=True).stdout
    return [dict(zip(('index','name','total','free','cap','driver','pci'),[v.strip() for v in line.split(',')])) for line in raw.strip().splitlines() if line.strip()]

try:
    phase(1,'Detecting platform')
    ensure(sys.platform=='linux' and platform.machine()=='x86_64', 'Only Linux x86_64 CUDA notebooks are supported')
    print('Platform:', 'Kaggle' if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else 'Colab' if 'google.colab' in sys.modules else 'Linux notebook')
    phase(2,'Detecting GPU(s)')
    gpus=memory(); ensure(gpus, 'No NVIDIA GPU visible; select a GPU runtime')
    smi=run(['nvidia-smi'],capture_output=True).stdout
    print(smi)
    try: print('Topology:',run(['nvidia-smi','topo','-m'],capture_output=True).stdout)
    except Exception: pass
    lib=ctypes.CDLL('libcuda.so.1'); count=ctypes.c_int()
    ensure(lib.cuInit(0)==0 and lib.cuDeviceGetCount(ctypes.byref(count))==0 and count.value>=1,'CUDA driver initialization failed')
    print('CUDA usable devices:',count.value)
    gpus=gpus[:count.value]
    for g in gpus: print(f"GPU {g['index']}: {g['name']}; {g['total']} MiB total; {g['free']} MiB free; SM {g['cap']}; PCI {g['pci']}")
    ensure(any(int(g['free'])>=9500 for g in gpus), 'Insufficient free VRAM for safe PQ2_0 GPU offload')
    print('CUDA driver capability:',re.search(r'CUDA Version:\s*([\d.]+)',smi).group(1) if 'CUDA Version:' in smi else 'unknown')
    phase(3,'Preparing PrismML runtime')
    if not (DEMO/'.git').exists(): run(['git','clone','--depth','1','https://github.com/PrismML-Eng/Bonsai-demo.git',str(DEMO)])
    run(['sh',str(DEMO/'scripts/download_binaries.sh')],cwd=DEMO)
    binary=DEMO/'bin/cuda/llama-server'
    ensure(binary.is_file() and os.access(binary,os.X_OK),'Official PrismML CUDA binary missing; refusing CPU fallback')
    env=os.environ.copy(); env['LD_LIBRARY_PATH']=str(binary.parent)+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    help_result=run([str(binary),'--help'],env=env,capture_output=True)
    helptext=help_result.stdout+help_result.stderr
    ensure('--api-key' in helptext and '--alias' in helptext and '--split-mode' in helptext, 'PrismML binary lacks required API/security flags')
    print('Runtime:',binary,'\nRelease:',(binary.parent/'.llama_release').read_text().strip())
    print('Version:',run([str(binary),'--version'],env=env,capture_output=True).stdout.strip()[:300])
    phase(4,'Preparing Bonsai 2 27B')
    install('huggingface_hub','huggingface_hub[hf_xet]')
    install('gguf','gguf')
    from huggingface_hub import HfApi, hf_hub_download
    from gguf import GGUFReader
    info=HfApi().model_info(REPO,files_metadata=True)
    sibling=next((x for x in info.siblings if x.rfilename==FILE),None)
    ensure(sibling is not None and sibling.lfs and sibling.lfs.get('sha256'), 'Official model SHA256 unavailable; cannot verify')
    expected=sibling.lfs['sha256']
    model=pathlib.Path(hf_hub_download(REPO,FILE,local_dir=str(ROOT/'models'),token=os.environ.get('HF_TOKEN') or os.environ.get('BONSAI_TOKEN')))
    ensure(model.stat().st_size>6_000_000_000,'GGUF is incomplete or implausibly small')
    print('Verifying official SHA256 (may take a minute)...',flush=True)
    digest=hashlib.sha256()
    with model.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): digest.update(chunk)
    ensure(digest.hexdigest()==expected,'Official model checksum mismatch: aborting')
    reader=GGUFReader(str(model),mode='r')
    def field(name):
        f=reader.get_field(name)
        return bytes(f.parts[-1]).decode('utf-8',errors='replace') if f else ''
    meta={k:field(k) for k in ('general.name','general.architecture','general.size_label','general.file_type')}
    ensure('bonsai' in meta['general.name'].lower() and ('2' in meta['general.name'] or '2' in meta['general.architecture']) and ('27' in meta['general.name'] or '27' in meta['general.size_label']),f'Unexpected GGUF metadata: {meta}')
    print('## MODEL VERIFICATION\nFamily: Bonsai 2\nParameters: ~27B\nFormat: PQ2_0\nPath:',model,'\nMetadata:',meta,'\nIntegrity: PASS')
    del reader
    phase(5,'Selecting optimized configuration')
    def supported(flag): return re.search(r'(?<!\w)'+re.escape(flag)+r'(?![\w-])',helptext) is not None
    context=16384 if min(int(g['free']) for g in gpus)>=11500 else 8192
    base=['-m',str(model),'--alias',ALIAS,'--host','127.0.0.1','-ngl','999','-c',str(context)]
    for flag,value in [('-np','1'),('-b','256'),('-ub','64'),('-fa','on'),('--temp','1.0'),('--top-p','0.95'),('--top-k','20'),('--min-p','0.05')]:
        if supported(flag): base += [flag,value]
    if supported('--jinja'): base+=['--jinja']
    # Never enable speculative decoding: upstream has no official Bonsai 2 drafter.
    key=os.environ.get('BONSAI_API_KEY') or (json.loads(STATE.read_text()).get('key') if STATE.exists() else None) or secrets.token_urlsafe(48)
    ensure(len(key)>=32,'BONSAI_API_KEY must have at least 32 characters')
    def start(dual=False):
        port=free_port(); args=[str(binary)]+base+['--port',str(port),'--api-key',key]
        if dual: args+=['--split-mode','layer','--main-gpu','0']
        else: args+=['--split-mode','none','--main-gpu','0']
        log=open(ROOT/('dual.log' if dual else 'single.log'),'w')
        e=env.copy(); e['CUDA_VISIBLE_DEVICES']='0,1' if dual else '0'
        proc=subprocess.Popen(args,cwd=DEMO,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close()
        for _ in range(360):
            if proc.poll() is not None: raise RuntimeError('PrismML server exited: '+(ROOT/('dual.log' if dual else 'single.log')).read_text()[-3000:])
            status,body=get(f'http://127.0.0.1:{port}/health',key,timeout=3)
            if status==200 and isinstance(body,dict) and body.get('status')=='ok': return proc,port
            time.sleep(2)
        proc.terminate(); raise RuntimeError('Model startup timed out; check '+str(ROOT/('dual.log' if dual else 'single.log')))
    def benchmark(port):
        prompt='Explain the following Python function and suggest two concrete improvements: '+('def search(items, target):\n    for index, value in enumerate(items):\n        if value == target: return index\n    return -1\n')*35
        url=f'http://127.0.0.1:{port}/v1/chat/completions'
        body={'model':ALIAS,'messages':[{'role':'user','content':prompt}],'max_tokens':96,'temperature':1.0,'stream':True,'stream_options':{'include_usage':True}}
        req=urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        t0=time.monotonic(); first=None; last=None; usage={}; chunks=[]
        with urllib.request.urlopen(req,timeout=240) as resp:
            for line in resp:
                if not line.startswith(b'data: '): continue
                if line.strip()==b'data: [DONE]': break
                event=json.loads(line[6:]); usage=event.get('usage') or usage
                if event.get('choices'):
                    delta=event['choices'][0].get('delta',{})
                    if delta.get('content') or delta.get('reasoning_content'):
                        first=first or time.monotonic(); last=time.monotonic(); chunks.append(delta)
        ensure(first and chunks and usage.get('completion_tokens',0)>1 and usage.get('prompt_tokens',0)>0,'Benchmark did not produce real streamed tokens/usage')
        # Server timing fields are preferred: wall clock TTFT includes both prefill and network latency.
        timings=usage.get('timings',{})
        input_rate=timings.get('prompt_per_second') or (usage['prompt_tokens']/(first-t0))
        output_rate=timings.get('predicted_per_second') or ((usage['completion_tokens']-1)/max(last-first,0.001))
        return dict(input=float(input_rate),output=float(output_rate),ttft=(first-t0)*1000,tokens=usage)
    def used(): return [int(x['total'])-int(x['free']) for x in memory()]
    phase(6,'Starting inference server')
    proc,port=start(); single=benchmark(port); single_used=used(); print('Single GPU:',single,'VRAM MiB:',single_used,flush=True)
    dual_result=None
    if len(gpus)>=2 and all(int(x['free'])>=9500 for x in gpus[:2]) and supported('--main-gpu'):
        proc.terminate(); proc.wait(timeout=30)
        try:
            proc,port=start(True); dual_result=benchmark(port); dual_used=used()
            ensure(dual_used[0]>1000 and dual_used[1]>1000,'Second GPU did not receive model work')
            print('Dual GPU:',dual_result,'VRAM MiB:',dual_used,flush=True)
        except Exception as exc:
            print('Dual GPU unavailable:',exc)
            if 'proc' in locals() and proc.poll() is None: proc.terminate(); proc.wait(timeout=30)
            dual_result=None
        if dual_result is None or dual_result['output']<=single['output']*1.03:
            if proc.poll() is None: proc.terminate(); proc.wait(timeout=30)
            proc,port=start(); selected='Single GPU (measured faster or within 3% of dual)'; result=benchmark(port)
        else: selected='Dual GPU layer split (measured >3% faster)'; result=dual_result
    else: selected='Single GPU'; result=single
    print('Selected:',selected)
    phase(7,'Running real API tests')
    local=f'http://127.0.0.1:{port}'
    status,health=get(local+'/health',key); ensure(status==200 and health.get('status')=='ok','Health check failed')
    status,models=get(local+'/v1/models',key); ensure(status==200 and any(x['id']==ALIAS for x in models['data']),'Model alias missing')
    status,denied=get(local+'/v1/models','invalid-'+secrets.token_hex(16)); ensure(status in (401,403),'Invalid key was accepted')
    status,denied=get(local+'/v1/models'); ensure(status in (401,403),'Unauthenticated API was accepted')
    request={'model':ALIAS,'messages':[{'role':'system','content':'Respond briefly.'},{'role':'user','content':'Say hello.'}],'max_tokens':128}
    status,answer=get(local+'/v1/chat/completions',key,request,timeout=240)
    ensure(status==200 and answer.get('model')==ALIAS and answer.get('choices'),'Real chat or model-name check failed: '+str(answer)[:500])
    status,bad=get(local+'/v1/chat/completions',key,{**request,'model':'not-bonsai'},timeout=30)
    ensure(status>=400,'Unknown model alias was accepted')
    result=benchmark(port) # also verifies streaming on final server
    print('Health, models, chat, streaming, model validation and authorization: PASS')
    phase(8,'Starting public tunnel')
    tunnel=ROOT/'cloudflared'
    if not tunnel.exists():
        url='https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64'
        urllib.request.urlretrieve(url,str(tunnel)); tunnel.chmod(0o700)
    tunnel_log=ROOT/'tunnel.log'
    existing=json.loads(STATE.read_text()) if STATE.exists() else {}
    tunnel_proc=None
    if existing.get('port')==port and existing.get('tunnel_pid'):
        try: os.kill(existing['tunnel_pid'],0); public=existing['public']
        except ProcessLookupError: public=None
    else: public=None
    if not public:
        with tunnel_log.open('w') as log:
            tunnel_proc=subprocess.Popen([str(tunnel),'tunnel','--url',local,'--no-autoupdate'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        public=None
        for _ in range(90):
            if tunnel_proc.poll() is not None: break
            match=re.search(r'https://[a-z0-9-]+\.trycloudflare\.com',tunnel_log.read_text())
            if match: public=match.group(0); break
            time.sleep(2)
    ensure(public,'Cloudflare tunnel startup failed: '+tunnel_log.read_text()[-1500:])
    status,remote=get(public+'/v1/models',key,timeout=30)
    ensure(status==200 and any(x['id']==ALIAS for x in remote['data']),'Remote tunnel API verification failed')
    STATE.write_text(json.dumps({'key':key,'port':port,'server_pid':proc.pid,'tunnel_pid':tunnel_proc.pid if tunnel_proc else existing['tunnel_pid'],'public':public}))
    os.chmod(STATE,0o600)
    print('\n'+'='*54+'\nTERNARY BONSAI 2 27B — READY\n'+'='*54)
    print('GPU:',', '.join(x['name'] for x in gpus),'\nBackend: PrismML CUDA\nSelected:',selected)
    print('Context:',context,'\nInput tokens/sec:',round(result['input'],2),'\nDecode tokens/sec:',round(result['output'],2),'\nTTFT ms:',round(result['ttft'],1),'\nGPU VRAM used MiB:',used())
    print('MODEL:',ALIAS,'\nHEALTH: OK\nAPI BASE URL:',public+'/v1\nAPI KEY:',key)
    print('\nPython example:\nfrom openai import OpenAI\nclient = OpenAI(base_url='+repr(public+'/v1')+', api_key='+repr(key)+')\nresponse = client.chat.completions.create(model='+repr(ALIAS)+', messages=[{"role":"user","content":"Write a C++ function that reverses a string."}])\nprint(response.choices[0].message.content)')
    print('\ncurl example:\ncurl -H "Authorization: Bearer '+key+'" -H "Content-Type: application/json" -d '+repr(json.dumps(request))+' '+public+'/v1/chat/completions')
except Exception as exc:
    print('\nDEPLOYMENT FAILED:',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
    raise
