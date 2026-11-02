"""Isolated review extension; original frozen viewer stays available."""
from pathlib import Path
import sys,re,argparse
from urllib.parse import urlparse
from http.server import ThreadingHTTPServer
import server
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from origami_bench.diagram import layer_review
class Handler(server.Handler):
 def do_GET(self):
  route=urlparse(self.path).path
  try:
   extra_out=ROOT/'outputs/layer-paths-v14-real-extension';extra_data=ROOT/'data/document-layers-v14-real-extension'
   runtime_out=ROOT/'outputs/layer-paths-v14-runtime-extension'
   v15_out=ROOT/'outputs/layer-paths-v15';v15_data=ROOT/'data/document-layers-v15'
   v16_out=ROOT/'outputs/layer-paths-v16';v16_data=ROOT/'data/document-layers-v16'
   v17_out=ROOT/'outputs/layer-paths-v17';v17_data=ROOT/'data/document-layers-v17'
   if route=='/api/layers-v17':return self.json(layer_review.summary(v17_out,v17_data))
   v17=re.fullmatch(r'/api/layers-v17/run/([A-Za-z0-9_-]+)',route)
   if v17:return self.json(layer_review.replay(v17[1],v17_out,v17_data,'/layer-v17-assets/'))
   v17_asset=re.fullmatch(r'/layer-v17-assets/([A-Za-z0-9_-]+)/(final|observation-[0-9]{3}|document-view-[0-9]{3})[.]png',route)
   if v17_asset:
    body=(layer_review.folder(v17_asset[1],v17_out)/(v17_asset[2]+'.png')).read_bytes();self.send_response(200);self.send_header('Content-Type','image/png');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
   if route=='/api/layers-v16':return self.json(layer_review.summary(v16_out,v16_data))
   v16=re.fullmatch(r'/api/layers-v16/run/([A-Za-z0-9_-]+)',route)
   if v16:return self.json(layer_review.replay(v16[1],v16_out,v16_data,'/layer-v16-assets/'))
   v16_asset=re.fullmatch(r'/layer-v16-assets/([A-Za-z0-9_-]+)/(final|observation-[0-9]{3}|document-view-[0-9]{3})[.]png',route)
   if v16_asset:
    body=(layer_review.folder(v16_asset[1],v16_out)/(v16_asset[2]+'.png')).read_bytes();self.send_response(200);self.send_header('Content-Type','image/png');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
   if route=='/api/layers-v15':return self.json(layer_review.summary(v15_out,v15_data))
   v15=re.fullmatch(r'/api/layers-v15/run/([A-Za-z0-9_-]+)',route)
   if v15:return self.json(layer_review.replay(v15[1],v15_out,v15_data,'/layer-v15-assets/'))
   v15_asset=re.fullmatch(r'/layer-v15-assets/([A-Za-z0-9_-]+)/(final|observation-[0-9]{3}|document-view-[0-9]{3})[.]png',route)
   if v15_asset:
    body=(layer_review.folder(v15_asset[1],v15_out)/(v15_asset[2]+'.png')).read_bytes();self.send_response(200);self.send_header('Content-Type','image/png');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
   if route=='/api/layers-runtime':return self.json(layer_review.summary(runtime_out,extra_data))
   runtime=re.fullmatch(r'/api/layers-runtime/run/([A-Za-z0-9_-]+)',route)
   if runtime:return self.json(layer_review.replay(runtime[1],runtime_out,extra_data,'/layer-runtime-assets/'))
   runtime_asset=re.fullmatch(r'/layer-runtime-assets/([A-Za-z0-9_-]+)/(final|observation-[0-9]{3}|document-view-[0-9]{3})[.]png',route)
   if runtime_asset:
    body=(layer_review.folder(runtime_asset[1],runtime_out)/(runtime_asset[2]+'.png')).read_bytes();self.send_response(200);self.send_header('Content-Type','image/png');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
   if route=='/api/layers-real':return self.json(layer_review.summary(extra_out,extra_data))
   extra=re.fullmatch(r'/api/layers-real/run/([A-Za-z0-9_-]+)',route)
   if extra:return self.json(layer_review.replay(extra[1],extra_out,extra_data,'/layer-real-assets/'))
   if route=='/api/layers':return self.json(layer_review.summary())
   match=re.fullmatch(r'/api/layers/run/([A-Za-z0-9_-]+)',route)
   if match:return self.json(layer_review.replay(match[1]))
   match=re.fullmatch(r'/layer-assets/([A-Za-z0-9_-]+)/(final|observation-[0-9]{3}|document-view-[0-9]{3})[.]png',route)
   extra=re.fullmatch(r'/layer-real-assets/([A-Za-z0-9_-]+)/(final|observation-[0-9]{3}|document-view-[0-9]{3})[.]png',route)
   if extra:
    p=layer_review.folder(extra[1],extra_out)/(extra[2]+'.png');body=p.read_bytes();self.send_response(200);self.send_header('Content-Type','image/png');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
   study=re.fullmatch(r'/layer-study-assets/(original|real|runtime)/(comparison|reference-steps|reference-sections)[.]png',route)
   if study:
    base=runtime_out if study[1]=='runtime' else extra_out if study[1]=='real' else ROOT/'outputs/layer-paths-v14';body=(base/(study[2]+'.png')).read_bytes();self.send_response(200);self.send_header('Content-Type','image/png');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
   if route in ('/layers','/layers-real','/layers-runtime','/layers-v15','/layers-v16','/layers-v17'):p=Path(__file__).with_name('layers.html');kind='text/html; charset=utf-8'
   elif match:p=layer_review.folder(match[1])/(match[2]+'.png');kind='image/png'
   elif re.fullmatch(r'/layer-diagrams/O[0-9]{2}[.]png',route):p=ROOT/'data/document-layers-v14/public/diagrams'/Path(route).name;kind='image/png'
   elif route.startswith('/layer-diagrams/'):return self.send_error(404)
   elif route.startswith(('/api/layers','/layer-assets/')):return self.send_error(404)
   else:return super().do_GET()
   body=p.read_bytes();self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
  except (ValueError,FileNotFoundError,KeyError):self.send_error(404)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8770);a=p.parse_args();print(f'Layer review http://127.0.0.1:{a.port}/layers',flush=True);ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
