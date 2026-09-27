"""Small stdlib same-origin web server and streaming proxy for Bonsai-Kit."""
from __future__ import annotations
import json, mimetypes, os, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from .client import BonsaiClient
from .errors import BonsaiAPIError, BonsaiError, StreamInterrupted, TransportError
from ._meta import VERSION
from .config import save_config

DIST = Path(__file__).parent / 'webui' / 'dist'
class _State:
    def __init__(self, endpoint, key, model):
        self.endpoint, self.key, self.model = endpoint, key or '', model
        self.client = BonsaiClient(endpoint, self.key, model=model, retries=1)
        self.streams, self.lock = {}, threading.RLock()

def serve(host='127.0.0.1', port=0, endpoint=None, key=None, model=None):
    if not DIST.is_dir():
        raise RuntimeError('web UI is not built; run `npm install && npm run build` in bonsai_chat/webui')
    state = _State(endpoint or os.environ.get('BONSAI_BASE_URL',''), key or os.environ.get('BONSAI_API_KEY',''), model)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args): pass
        def send_json(self, obj, status=200):
            raw=json.dumps(obj).encode(); self.send_response(status); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
        def body(self): return json.loads(self.rfile.read(int(self.headers.get('Content-Length','0')) or 0) or b'{}')
        def do_GET(self):
            if self.path == '/api/health':
                t=time.monotonic(); h=state.client.health(); self.send_json({'ok':h is not None,'endpoint':state.client.base_url,'model':state.client.model,'latency_ms':round((time.monotonic()-t)*1000,2) if h is not None else None}); return
            if self.path == '/api/models':
                try: self.send_json(state.client.models())
                except Exception as e: self.send_json({'error':str(e),'kind':_kind(e)},502)
                return
            if self.path == '/api/capabilities': self.send_json(state.client.caps.to_dict()); return
            if self.path == '/api/version': self.send_json({'version':VERSION}); return
            if self.path == '/api/config': self.send_json({'endpoint':state.client.base_url,'model':state.client.model}); return
            rel='index.html' if self.path in ('/','') else self.path.removeprefix('/ui/')
            p=(DIST/rel).resolve()
            if not str(p).startswith(str(DIST.resolve())) or not p.is_file(): self.send_error(404); return
            data=p.read_bytes(); self.send_response(200); self.send_header('Content-Type',mimetypes.guess_type(str(p))[0] or 'application/octet-stream'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_PUT(self):
            if self.path != '/api/config': self.send_error(404); return
            d=self.body(); state.endpoint=d.get('endpoint',state.client.base_url); state.key=d.get('api_key',state.key); state.model=d.get('model',state.client.model); state.client=BonsaiClient(state.endpoint,state.key,model=state.model,retries=1); self.send_json({'ok':True})
        def do_POST(self):
            if self.path == '/api/stop':
                sid=self.body().get('stream_id'); s=state.streams.get(sid)
                if s: s.close()
                self.send_json({'ok':True}); return
            if self.path != '/api/chat': self.send_error(404); return
            d=self.body(); sid=d.get('stream_id') or str(time.time_ns())
            try: stream=state.client.stream_chat(d.get('messages',[]),**(d.get('sampling') or {}))
            except Exception as e: self.send_json({'error':str(e),'kind':_kind(e)},502); return
            with state.lock: state.streams[sid]=stream
            try:
                self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.send_header('Cache-Control','no-cache'); self.end_headers()
                for ev in stream:
                    typ=ev.get('kind','delta'); payload=json.dumps(ev,separators=(',',':')); self.wfile.write(('event: '+typ+'\ndata: '+payload+'\n\n').encode()); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError): stream.close()
            except Exception as e:
                try: self.wfile.write(('event: error\ndata: '+json.dumps({'kind':_kind(e),'error':str(e)})+'\n\n').encode()); self.wfile.flush()
                except OSError: pass
            finally:
                with state.lock: state.streams.pop(sid,None)
                stream.close()
    httpd=ThreadingHTTPServer((host,port),Handler); print(f'Bonsai web UI: http://{host}:{httpd.server_port}',flush=True); return httpd

def _kind(e):
    if isinstance(e,BonsaiAPIError) and e.status in (401,403): return 'unauthorized'
    if isinstance(e,StreamInterrupted): return 'mid_stream'
    if isinstance(e,(TimeoutError,TimeoutError)): return 'stall'
    return 'connection' if isinstance(e,(BonsaiError,TransportError,OSError)) else 'remote'
