"""Create a standalone interactive HTML replay from a completed run."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
state=json.loads((a.run/'final-state.json').read_text())
report=json.loads((a.run/'report.json').read_text())
v=np.load(a.run/'trajectory.npz')['vertices']
indices=np.unique(np.linspace(0,len(v)-1,min(len(v),240),dtype=int))
payload=json.dumps(dict(vertices=np.round(v[indices],7).tolist(),frames=indices.tolist(),faces=state['faces'],report=report)).replace('</','<\\/')
three=(ROOT/'viewer/benchmark/vendor/three.min.js').read_text()
orbit=(ROOT/'viewer/benchmark/vendor/OrbitControls.js').read_text()
html='''<!doctype html><meta charset="utf-8"><title>Origami replay</title>
<style>body{margin:0;font:16px system-ui;background:#edf1f6}#info{position:absolute;left:18px;top:12px;max-width:420px;background:white;padding:16px;border-radius:10px}input{width:300px}pre{white-space:pre-wrap;max-height:30vh;overflow:auto;font-size:12px}</style>
<div id="info"><b>Origami benchmark · physical replay</b><p>Drag to rotate · Scroll to zoom</p><input id="frame" type="range" min="0" value="0"><p id="label"></p><details><summary>Recorded report</summary><pre id="report"></pre></details></div>
<script>'''+three+'\n'+orbit+'''</script><script>
const data='''+payload+''';
const scene=new THREE.Scene(), camera=new THREE.PerspectiveCamera(45,innerWidth/innerHeight,.001,10);
camera.position.set(.25,-.25,.3);camera.up.set(0,0,1);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setSize(innerWidth,innerHeight);renderer.setClearColor(0xedf1f6);document.body.appendChild(renderer.domElement);
const controls=new THREE.OrbitControls(camera,renderer.domElement);controls.target.set(0,0,0);
scene.add(new THREE.AmbientLight(0x888888));const lamp=new THREE.DirectionalLight(0xffffff,.8);lamp.position.set(0,-1,2);scene.add(lamp);
const geometry=new THREE.Geometry();data.vertices[0].forEach(v=>geometry.vertices.push(new THREE.Vector3(...v)));data.faces.forEach(f=>geometry.faces.push(new THREE.Face3(...f)));
const paper=new THREE.Mesh(geometry,new THREE.MeshPhongMaterial({color:0x83bfe0,side:THREE.DoubleSide}));scene.add(paper);
const slider=document.getElementById('frame');slider.max=data.vertices.length-1;
document.getElementById('report').textContent=JSON.stringify(data.report,null,2);
function show(){const i=+slider.value;data.vertices[i].forEach((v,j)=>geometry.vertices[j].set(...v));geometry.verticesNeedUpdate=true;geometry.computeFaceNormals();geometry.computeVertexNormals();document.getElementById('label').textContent='Frame '+data.frames[i]+' / '+data.frames[data.frames.length-1]+' · recorded result: '+data.report.status;}
slider.oninput=show;show();function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera);}animate();onresize=()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);};
</script>'''
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(html)
print(a.output.resolve())
