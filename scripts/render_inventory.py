"""Rebuild the public audit page from inventory.json; no private source tree needed."""
from pathlib import Path
import html
import json

ROOT=Path(__file__).resolve().parents[1]

def main():
    document=json.loads((ROOT/'audit/inventory.json').read_text())
    rows=document['records'];counts=document['counts'];published=document['published_counts']
    labels={'accepted':'已验收','needs_repair':'待修复 / 待补齐','invalid':'无效数据'}
    tracks={'step_panels':'逐步图解','prepared_layers':'多层中间态','external_documents':'整页原站图','external_prepared':'原站图 + 中间态','external_manual':'需自备图源','diagnostic':'仅诊断'}
    translations={
        'frozen data + recomputed semantic/geometry/physical checks':'冻结数据校验和参考轨迹重新评分通过',
        'reference execution did not pass fixed grading':'参考路径未通过固定评分规则；失败记录保留',
        'Placeholder/unsupported task specification is not an executable scored benchmark':'占位数据或未支持操作，不能作为有效评分题',
        'No completed reference plus complete diagram/goal/checkpoint bundle':'缺少完整图解、目标/检查点或完成的参考路径',
        'physical path and legacy score pass; ordered rotation/pose grading needs repair':'物理路径及旧评分通过；旋转、姿态和步骤顺序评分待修复',
        'Physical reference passed, but pose/ordered-path grading ambiguity needs repair':'物理路径通过，但姿态和步骤顺序评分仍有歧义',
        'Historical physics passed; source PDF returns HTTP 403. Matching local source required; excluded from default suites.':'历史物理验收通过，但原站返回 403；须自备匹配 PDF，默认不计分',
        'Source intake only; an entire work is not a qualified executable task. Qualified sub-tasks are listed separately.':'仅有作品题源；整件作品未完成验收，通过的子任务另列',
    }
    rendered=[]
    for row in rows:
        title=html.escape(row['task_id']+' · '+row['title'])
        if row.get('diagram_path'):
            title+=' <a href="../'+html.escape(row['diagram_path'])+'">查看图解</a>'
        elif row.get('source_url'):
            title+=' <a href="'+html.escape(row['source_url'])+'">原站图解</a>'
        if row.get('public_instruction'):
            title+='<details><summary>题目范围</summary>'+html.escape(row['public_instruction'])+'</details>'
        if row.get('release_reference_evidence'):
            title+=' <a href="../'+html.escape(row['release_reference_evidence'])+'/report.json">本轮验收</a>'
        if row.get('replay_path'):
            title+=' <a href="../'+html.escape(row['replay_path'])+'">3D 回放</a>'
        availability='可运行 · '+tracks.get(row.get('public_track'),'') if row['runnable'] else '旧版 → '+row['superseded_by'] if row.get('superseded_by') else '未入评分集'
        cells=[html.escape(row['dataset']),title,labels[row['status']],html.escape(availability),html.escape(translations.get(row['reason'],row['reason']))]
        rendered.append('<tr data-status="'+row['status']+'">'+''.join('<td>'+x+'</td>' for x in cells)+'</tr>')
    external=published['accepted']-published['bundled_accepted']
    blocked=published['external_source_required']-external
    header=f"{published['bundled_accepted']} 道自带素材的已验收题 + {external} 道原站获取已验证题 + {blocked} 道原站 403 待修复题 + {published['diagnostic']} 道可运行诊断题。"
    scope=f"下表为 {len(rows)} 条历史版本 / 题源记录，含旧版本重复项：{counts['accepted']} 条已验收记录、{counts['needs_repair']} 条待修复或待补齐记录、{counts['invalid']} 条无效记录。"
    page='''<!doctype html><meta charset="utf-8"><title>Origami inventory</title>
<style>body{font:15px system-ui;margin:30px;color:#172b43}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}th{position:sticky;top:0;background:#edf4ff}input,select{font:inherit;padding:8px;margin:8px}input{width:300px}tr[data-status=invalid]{background:#fff0ed}</style>
<h1>折纸 benchmark 全量盘点</h1><p>'''+header+'''</p><p>早期逐步裁图与后期整页读图必须分轨报告；诊断题不得混入正式分数。</p><p>'''+scope+'''</p><p id="count"></p>
<input id="q" placeholder="搜索数据集 / 题号 / 原因"><select id="s"><option value="">全部</option><option value="accepted">已验收</option><option value="needs_repair">待修复</option><option value="invalid">无效</option></select>
<table><thead><tr><th>数据集</th><th>题号 / 图解</th><th>分类</th><th>公开运行</th><th>依据 / 阻塞</th></tr></thead><tbody>'''+''.join(rendered)+'''</tbody></table>
<script>const q=document.getElementById('q'),s=document.getElementById('s'),rows=[...document.querySelectorAll('tbody tr')];function filter(){rows.forEach(r=>r.hidden=!(r.textContent.toLowerCase().includes(q.value.toLowerCase())&&(!s.value||r.dataset.status===s.value)));document.getElementById('count').textContent='当前显示 '+rows.filter(r=>!r.hidden).length+' / '+rows.length+' 条记录'}q.oninput=s.onchange=filter;filter();</script>'''
    (ROOT/'audit/inventory.html').write_text(page)

if __name__=='__main__':main()
