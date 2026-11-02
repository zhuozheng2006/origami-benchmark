"""Acquire original source PDFs locally; never redistribute pages in this repository."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--source',action='append',help='Source ID, repeatable; default sources currently downloadable');p.add_argument('--pdf-root',type=Path,help='Optional local folder containing SOURCE/source.pdf');a=p.parse_args()
import pypdfium2 as pdfium
sources=json.loads((ROOT/'data/external-qualified/source-manifest.json').read_text())
for source in sources:
 if a.source and source['id'] not in a.source:continue
 if not a.source and not a.pdf_root and source.get('automatic_download') is False:
  print(source['id'],'source currently unavailable online; use --pdf-root with a matching local PDF');continue
 sid=source['id'];dest=ROOT/'.local/diagram-sources-tiered-v1'/sid;dest.mkdir(parents=True,exist_ok=True)
 if a.pdf_root:raw=(a.pdf_root/sid/'source.pdf').read_bytes()
 else:
  with urllib.request.urlopen(source['url'],timeout=60) as response:raw=response.read(50_000_001)
 if len(raw)>50_000_000 or hashlib.sha256(raw).hexdigest()!=source['pdf_sha256']:raise ValueError('Source differs from frozen PDF: '+sid)
 (dest/'source.pdf').write_bytes(raw);doc=pdfium.PdfDocument(raw)
 if len(doc)!=len(source['pages']):raise ValueError('Page count differs: '+sid)
 for page in source['pages']:
  pg=doc[page['page']-1];im=pg.render(scale=1280/max(pg.get_size())).to_pil().convert('RGB')
  if list(im.size)!=page['size'] or hashlib.sha256(im.tobytes()).hexdigest()!=page['pixel_sha256']:
   raise ValueError('Rendered pixels differ; use pinned PDFium/Pillow versions, do not silently change model inputs: '+sid)
  im.save(dest/f"page-{page['page']:03}.png");pg.close()
 doc.close();print(sid,'verified PDF and all page pixels')
