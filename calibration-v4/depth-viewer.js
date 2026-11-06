/* Local Three.js r71 renderer, reused from the project's established viewer.
 * Rendering changes only: source vertices, winding, spacing and topology stay
 * unchanged. Opaque front/back materials share a real WebGL depth buffer.
 */
(function (global) {
  'use strict';

  function makeGeometry(THREE, vertices, faces) {
    const geometry = new THREE.Geometry();
    for (const p of vertices) geometry.vertices.push(new THREE.Vector3(p[0], p[1], p[2]));
    for (const f of faces) geometry.faces.push(new THREE.Face3(f[0], f[1], f[2]));
    geometry.computeFaceNormals();
    geometry.computeBoundingSphere();
    return geometry;
  }

  function updateGeometry(geometry, vertices) {
    if (geometry.vertices.length !== vertices.length) throw Error('Saved mesh topology changed');
    for (let i = 0; i < vertices.length; i++) {
      const p = vertices[i];
      geometry.vertices[i].set(p[0], p[1], p[2]);
    }
    geometry.verticesNeedUpdate = true;
    geometry.computeFaceNormals();
    geometry.normalsNeedUpdate = true;
    geometry.computeBoundingSphere();
  }

  function bounds(frames) {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const frame of frames) for (const p of frame) for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]);
    }
    return {center: lo.map((v, k) => (v + hi[k]) / 2), span: Math.max(.02, ...hi.map((v, k) => v - lo[k]))};
  }

  class DepthViewer {
    constructor(canvas, onStatus) {
      const THREE = global.THREE;
      if (!THREE || !THREE.OrbitControls) throw Error('Local Three.js / OrbitControls did not load');
      this.THREE = THREE;
      this.canvas = canvas;
      this.onStatus = onStatus || (() => {});
      this.renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: false, depth: true, preserveDrawingBuffer: true});
      this.renderer.setClearColor(0xedf2f7, 1);
      this.renderer.setPixelRatio(Math.min(global.devicePixelRatio || 1, 2));
      this.scene = new THREE.Scene();
      this.camera = new THREE.OrthographicCamera(-.15, .15, .15, -.15, .001, 3);
      this.camera.up.set(0, 0, 1);
      this.camera.position.set(.25, -.35, .3);
      this.controls = new THREE.OrbitControls(this.camera, canvas);
      this.controls.minZoom = .25;
      this.controls.maxZoom = 12;
      this.controls.addEventListener('change', () => this.render());
      this.data = null;
      this.frame = 0;
      this.paper = null;
      this.geometry = null;
      this.lines = [];
      this.viewSpan = .26;
      const gl = this.renderer.getContext();
      this.depthBits = gl.getParameter(gl.DEPTH_BITS);
      this.onStatus({renderer: 'WebGL', three_revision: THREE.REVISION, depth_bits: this.depthBits,
        geometry_offset: 0, coordinate_scale: 1, opaque_depth_test: true});
      if (this.depthBits < 16) throw Error('A usable WebGL depth buffer is unavailable');
      this.resize();
    }

    removePaper() {
      this.removeLines();
      if (this.paper) {
        this.scene.remove(this.paper);
        for (const child of this.paper.children) if (child.material) child.material.dispose();
        this.geometry.dispose();
      }
      this.paper = this.geometry = null;
    }

    removeLines() {
      for (const line of this.lines) {
        this.scene.remove(line); line.geometry.dispose(); line.material.dispose();
      }
      this.lines = [];
    }

    setData(data) {
      if (this.data === data) return;
      this.removePaper(); this.data = data;
      if (!data) { this.render(); return; }
      const THREE = this.THREE;
      this.geometry = makeGeometry(THREE, data.vertices[0], data.faces);
      this.paper = new THREE.Object3D();
      this.paper.name = 'actual-paper';
      const common = {depthTest: true, depthWrite: true, transparent: false, opacity: 1};
      this.paper.add(new THREE.Mesh(this.geometry, new THREE.MeshBasicMaterial(Object.assign({}, common,
        {color: 0x9cc4dd, side: THREE.FrontSide}))));
      this.paper.add(new THREE.Mesh(this.geometry, new THREE.MeshBasicMaterial(Object.assign({}, common,
        {color: 0xe9d5ac, side: THREE.BackSide}))));
      this.wire = new THREE.Mesh(this.geometry, new THREE.MeshBasicMaterial({color: 0x687f92,
        side: THREE.DoubleSide, wireframe: true, depthTest: true, depthWrite: false, transparent: false}));
      this.wire.visible = false;
      this.paper.add(this.wire);
      this.scene.add(this.paper);
      this.reset();
    }

    setFrame(index, wire, creases) {
      if (!this.data) return;
      this.frame = index;
      updateGeometry(this.geometry, this.data.vertices[index]);
      this.wire.visible = !!wire;
      this.removeLines();
      const history = this.data.crease_history;
      if (creases && history && history.frames[index]) {
        const THREE = this.THREE, rows = history.frames[index], vertices = this.data.vertices[index];
        for (const [sign, color] of [[1, 0xbf4654], [2, 0x286cc0], [3, 0x8c4fba]]) {
          const geometry = new THREE.Geometry();
          for (const [i, j, mask] of rows) if (mask === sign) {
            // Exact mesh endpoints; no artificial normal offsets or layer gaps.
            geometry.vertices.push(new THREE.Vector3(...vertices[i]), new THREE.Vector3(...vertices[j]));
          }
          if (!geometry.vertices.length) { geometry.dispose(); continue; }
          const material = new THREE.LineBasicMaterial({color, linewidth: 1,
            depthTest: true, depthWrite: false, transparent: false});
          const line = new THREE.Line(geometry, material, THREE.LinePieces);
          this.scene.add(line); this.lines.push(line);
        }
      }
      this.render();
    }

    reset() {
      if (!this.data) return;
      const b = bounds([this.data.vertices[0], this.data.vertices[this.data.vertices.length - 1], this.data.vertices[this.frame] || this.data.vertices[0]]);
      const THREE = this.THREE;
      const center = new THREE.Vector3(...b.center);
      this.viewSpan = b.span * 1.55;
      this.camera.zoom = 1;
      this.camera.near = Math.max(.00001, b.span * .01);
      this.camera.far = Math.max(1, b.span * 20);
      this.controls.target.copy(center);
      this.camera.position.copy(center.clone().add(new THREE.Vector3(.65, -1, .9).normalize().multiplyScalar(b.span * 2.4)));
      this.resize(); this.controls.update(); this.render();
    }

    resize() {
      const r = this.canvas.getBoundingClientRect(), width = Math.max(1, r.width), height = Math.max(1, r.height);
      const aspect = width / height;
      this.camera.left = -this.viewSpan * aspect / 2;
      this.camera.right = this.viewSpan * aspect / 2;
      this.camera.top = this.viewSpan / 2;
      this.camera.bottom = -this.viewSpan / 2;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(width, height, false);
      this.render();
    }

    render() {
      this.renderer.render(this.scene, this.camera);
      const r = this.canvas.getBoundingClientRect();
      this.onStatus({renderer: 'WebGL', three_revision: this.THREE.REVISION, depth_bits: this.depthBits,
        scale_20mm_pixels: .02 * r.height / this.viewSpan * this.camera.zoom,
        geometry_offset: 0, coordinate_scale: 1, opaque_depth_test: true});
    }

    tick() { this.controls.update(); }
  }

  global.OrigamiDepthViewer = DepthViewer;
  if (typeof module !== 'undefined' && module.exports) module.exports = {DepthViewer, makeGeometry, updateGeometry, bounds};
})(typeof window !== 'undefined' ? window : globalThis);
