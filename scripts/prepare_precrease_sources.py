"""Acquire and deterministically render a frozen real-source page subset.

PDFs/pages stay in an ignored local cache. Manifest hashes are over PDF bytes
and RGB pixel bytes, never over PNG encoding. No model or simulator is used.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def prepare(manifest_path, output_root, pdf_root=None, source_ids=None):
    import pypdfium2 as pdfium
    import PIL
    sources = json.loads(Path(manifest_path).read_text())
    results = []
    for source in sources:
        if source_ids and source['id'] not in source_ids:
            continue
        render = source['render']
        versions = dict(pypdfium2=importlib.metadata.version('pypdfium2'),
                        pdfium=str(pdfium.PDFIUM_INFO), pillow=PIL.__version__)
        if versions != render['versions']:
            raise ValueError(f"Renderer differs from frozen versions: {source['id']}: {versions}")
        if render['color_mode'] != 'RGB' or render['max_dimension'] <= 0:
            raise ValueError('Unsupported rendering contract')
        relative = Path(source['cache_root'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('cache_root must stay inside output_root')
        destination = Path(output_root) / relative
        if pdf_root:
            raw = (Path(pdf_root) / source['id'] / 'source.pdf').read_bytes()
        elif (destination / 'source.pdf').exists():
            raw = (destination / 'source.pdf').read_bytes()
        else:
            request = urllib.request.Request(source['url'], headers={
                'User-Agent': 'Mozilla/5.0 OrigamiBenchmarkSourceVerifier/1.0',
                'Accept': 'application/pdf'})
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read(50_000_001)
        if len(raw) > 50_000_000 or hashlib.sha256(raw).hexdigest() != source['pdf_sha256']:
            raise ValueError('Downloaded/local PDF differs from frozen source: ' + source['id'])
        destination.mkdir(parents=True, exist_ok=True)
        document = pdfium.PdfDocument(raw)
        if len(document) != source['pdf_page_count']:
            raise ValueError('Frozen PDF page count differs')
        page_results = []
        for target in source['pages']:
            page = document[target['page'] - 1]
            image = page.render(scale=render['max_dimension'] / max(page.get_size())).to_pil().convert('RGB')
            pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
            if list(image.size) != target['size'] or pixel_hash != target['pixel_sha256']:
                raise ValueError('Rendered RGB pixels differ: ' + source['id'])
            target_path = destination / f"page-{target['page']:03}.png"
            image.save(target_path)
            page_results.append(dict(page=target['page'], pixel_sha256=pixel_hash, path=str(target_path)))
            page.close()
        document.close()
        (destination / 'source.pdf').write_bytes(raw)
        results.append(dict(source=source['id'], pdf_sha256=source['pdf_sha256'], pages=page_results, verified=True))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/precrease-hard-v4/source-manifest.json')
    parser.add_argument('--output-root', type=Path, default=ROOT)
    parser.add_argument('--pdf-root', type=Path, help='Local SOURCE_ID/source.pdf files instead of downloading')
    parser.add_argument('--source', action='append')
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.output_root, args.pdf_root, args.source), indent=2))


if __name__ == '__main__':
    main()
