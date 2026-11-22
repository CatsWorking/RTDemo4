import { Renderer } from './engine.js?v=6';
import { Camera } from './camera.js?v=6';
import {
  PRESETS,
  DEFAULT_ENV,
  MATERIALS,
  MAX_OBJECTS,
  makeObject,
  objectLabel,
  remixObjects,
  clampObject,
  makeScenePayload,
  parseScenePayload,
} from './scene.js?v=6';
import {
  clamp,
  hexToRgb,
  smoothstep,
  normalize3,
  sub3,
  add3,
  mul3,
  dot3,
  len3,
  downloadBlob,
  formatClock,
  formatDuration,
  formatCount,
} from './utils.js?v=6';

const $ = (id) => document.getElementById(id);
const AXES = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
const GFX_LABELS = {
  volumetrics: 'Volumetric fog',
  bloomFx: 'Bloom',
  dof: 'Depth of field',
  shadows: 'Shadows',
  reflections: 'Floor reflections',
};

const QUALITY_PRESETS = {
  potato: { autoRes: true, scale: 0.35, bounces: 2, samples: 512, fps: 60, denoise: true, vol: 6, bloom: 0.2 },
  balanced: { autoRes: true, scale: 0.5, bounces: 3, samples: 1024, fps: 60, denoise: true, vol: 8, bloom: 0.3 },
  gtx1070: { autoRes: true, scale: 0.75, bounces: 3, samples: 2048, fps: 60, denoise: true, vol: 10, bloom: 0.35 },
  quality: { autoRes: false, scale: 1, bounces: 5, samples: 8192, fps: 0, denoise: true, vol: 16, bloom: 0.4 },
  cinematic: { autoRes: false, scale: 1.5, bounces: 6, samples: 16384, fps: 0, denoise: false, vol: 24, bloom: 0.45 },
};

function mixRgb(a, b, t) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

function computeEnvUniforms(env) {
  const t = (env.time / 24) * Math.PI * 2 - Math.PI / 2;
  const sunDir = normalize3([Math.cos(t), Math.sin(t), 0.3]);
  const elev = sunDir[1];
  const base = hexToRgb(env.sunColor);
  const day = smoothstep(-0.05, 0.4, elev);
  const horizonWarm = 1 - smoothstep(0.02, 0.34, Math.abs(elev));

  let sky = mixRgb([0.008, 0.012, 0.03], [0.16, 0.36, 0.78], day);
  let horizon = mixRgb([0.012, 0.018, 0.055], [0.7, 0.84, 1.0], day);
  sky = mixRgb(sky, [0.38, 0.2, 0.3], horizonWarm * 0.45);
  horizon = mixRgb(horizon, [1.0, 0.43, 0.16], horizonWarm * 0.72);

  const intensity = elev > 0 ? 0.3 + 2.15 * Math.pow(clamp(elev, 0, 1), 0.62) : 0;
  const sunTint = mixRgb([1.0, 0.4, 0.16], base, smoothstep(0.02, 0.3, elev));
  const sunColor = sunTint.map((c) => c * intensity);

  return {
    sunDir,
    sunColor,
    skyColor: sky,
    horizonColor: horizon,
    fogDensity: env.fog,
    fogAniso: env.aniso,
    fogHeight: env.fogHeight,
    fogGlow: env.fogGlow,
    floorColor: hexToRgb(env.floorColor),
  };
}

function cloneCamera(cam) {
  const c = new Camera({
    pos: [...cam.pos],
    yaw: cam.yaw,
    pitch: cam.pitch,
    fov: cam.fov,
    visible: cam.visible,
    color: cam.color,
  });
  c.aperture = cam.aperture || 0;
  c.focus = cam.focus || 6;
  return c;
}

function rayFromScreen(cam, x, y, w, h) {
  const ndcX = (x / w) * 2 - 1;
  const ndcY = 1 - (y / h) * 2;
  return cam.ray(ndcX, ndcY, w / h);
}

function rayPlane(ray, origin, normal) {
  const denom = dot3(ray.dir, normal);
  if (Math.abs(denom) < 1e-5) return null;
  const t = dot3(sub3(origin, ray.origin), normal) / denom;
  if (t < 0.01) return null;
  return add3(ray.origin, mul3(ray.dir, t));
}

function closestRayToAxis(ray, origin, axis, maxU) {
  const w0 = sub3(ray.origin, origin);
  const b = dot3(ray.dir, axis);
  const d = dot3(ray.dir, w0);
  const e = dot3(axis, w0);
  const denom = 1 - b * b;
  let u = Math.abs(denom) > 1e-6 ? (e - b * d) / denom : 0;
  if (maxU !== undefined) u = clamp(u, 0, maxU);
  const point = add3(origin, mul3(axis, u));
  const t = Math.max(0, dot3(sub3(point, ray.origin), ray.dir));
  const closest = add3(ray.origin, mul3(ray.dir, t));
  return { u, dist: len3(sub3(point, closest)) };
}

function rayObject(ray, o) {
  if (o.type === 1) {
    const r = o.scale[0] * 0.5;
    const oc = sub3(ray.origin, o.pos);
    const b = dot3(oc, ray.dir);
    const c = dot3(oc, oc) - r * r;
    const h = b * b - c;
    if (h < 0) return -1;
    const t = -b - Math.sqrt(h);
    return t > 0.01 ? t : -1;
  }
  const half = o.scale.map((v) => v * 0.5);
  let tmin = -Infinity;
  let tmax = Infinity;
  for (let i = 0; i < 3; i++) {
    if (Math.abs(ray.dir[i]) < 1e-8) {
      if (ray.origin[i] < o.pos[i] - half[i] || ray.origin[i] > o.pos[i] + half[i]) return -1;
      continue;
    }
    const inv = 1 / ray.dir[i];
    let t0 = (o.pos[i] - half[i] - ray.origin[i]) * inv;
    let t1 = (o.pos[i] + half[i] - ray.origin[i]) * inv;
    if (t0 > t1) [t0, t1] = [t1, t0];
    tmin = Math.max(tmin, t0);
    tmax = Math.min(tmax, t1);
  }
  if (tmax < tmin || tmax < 0.01) return -1;
  return tmin > 0.01 ? tmin : tmax;
}

class App {
  constructor() {
    this.canvas = $('view');
    this.engine = null;
    try {
      this.engine = new Renderer(this.canvas);
    } catch (err) {
      console.error(err);
      this.fatal(err.message || String(err));
      return;
    }
    if (this.engine.failed) {
      this.fatal(this.engine.failReason);
      return;
    }
    this.engine.onExportProgress = (p) => this.onExportProgress(p);
    this.engine.onExportDone = (r) => this.onExportDone(r);
    this.engine.onScreenshotReady = () => this.toast('Screenshot saved');
    this.canvas.addEventListener('webglcontextlost', (e) => {
      e.preventDefault();
      this.fatal('The WebGL context was lost. Reload the page to restart the renderer.');
    });

    this.player = new Camera({ pos: [0, 2, 7], yaw: -Math.PI / 2, fov: 90 });
    this.renderCam = null;
    this.objects = [];
    this.selected = -1;
    this.env = { ...DEFAULT_ENV };
    this.settings = {
      autoRes: true,
      resScale: 0.75,
      bounces: 3,
      maxSamples: 2048,
      fpsLimit: 60,
      denoise: true,
      outlines: true,
      volumetrics: true,
      bloomFx: true,
      dof: true,
      shadows: true,
      reflections: true,
    };
    this.moveSpeed = 6;
    this.sensitivity = 1;
    this.fly = false;
    this.pipEnabled = true;
    this.locked = false;
    this.keys = {};
    this.uiSync = false;
    this.exporting = false;
    this.sizeLink = true;
    this.qualityPreset = 'gtx1070';
    this.frameTimes = [];
    this.drawTimes = [];
    this.lastHud = 0;
    this.lastTime = performance.now();
    this.lastFrame = performance.now();
    this.gizmo = {
      hot: 0,
      dragging: false,
      axis: 0,
      startPos: [0, 0, 0],
      startU: 0,
      startPoint: null,
    };

    this.bindUI();
    this.bindInput();
    this.applyQuality('gtx1070', true);
    this.applyPreset('classic', true);
    this.syncAllUI();
    try {
      this.engine.render(this.buildFrameState(), performance.now());
      this.updateHud(performance.now());
    } catch (err) {
      console.error(err);
    }
    requestAnimationFrame(this.loop.bind(this));
  }

  fatal(message) {
    const el = $('error-screen');
    if (el) {
      $('error-detail').textContent = message || 'The renderer could not start.';
      el.classList.add('open');
    }
  }

  get viewCam() {
    return this.fly && this.renderCam ? this.renderCam : this.player;
  }

  get selectedObject() {
    return this.selected >= 0 && this.selected < this.objects.length ? this.objects[this.selected] : null;
  }

  // ---------------------------------------------------------------- UI setup
  bindUI() {
    $('preset-grid').addEventListener('click', (e) => {
      const btn = e.target.closest('.preset-btn');
      if (btn) this.applyPreset(btn.dataset.preset);
    });

    $('btn-add-cube').addEventListener('click', () => this.spawn(0));
    $('btn-add-sphere').addEventListener('click', () => this.spawn(1));
    $('btn-clear').addEventListener('click', () => this.clearScene());
    $('btn-remix').addEventListener('click', () => this.remix());

    $('btn-look').addEventListener('click', () => this.toggleLook());
    $('btn-shot').addEventListener('click', () => this.screenshot());
    $('btn-shot2').addEventListener('click', () => this.screenshot());
    $('btn-save').addEventListener('click', () => this.saveScene());
    $('btn-save2').addEventListener('click', () => this.saveScene());
    $('btn-load').addEventListener('click', () => $('file-load').click());
    $('btn-load2').addEventListener('click', () => $('file-load').click());
    $('file-load').addEventListener('change', (e) => this.loadSceneFile(e.target.files[0]));

    for (const id of ['btn-help', 'btn-about', 'btn-changelog']) {
      $(id).addEventListener('click', () => this.openModal(id.replace('btn-', 'modal-')));
    }
    document.querySelectorAll('.modal-close').forEach((b) =>
      b.addEventListener('click', () => this.closeModal(b.dataset.close)),
    );
    document.querySelectorAll('.modal').forEach((m) =>
      m.addEventListener('pointerdown', (e) => {
        if (e.target === m) this.closeModal(m.id);
      }),
    );

    document.querySelectorAll('.tab-btn').forEach((b) =>
      b.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach((x) => x.classList.toggle('active', x === b));
        document.querySelectorAll('.tab-panel').forEach((p) => p.classList.toggle('active', p.id === `tab-${b.dataset.tab}`));
      }),
    );

    // object inspector
    $('sel-material').addEventListener('change', () => {
      const o = this.selectedObject;
      if (!o) return;
      o.mat = Number($('sel-material').value);
      o.prop = MATERIALS[o.mat].def;
      this.syncInspector();
      this.invalidate();
    });
    $('sel-color').addEventListener('input', () => {
      const o = this.selectedObject;
      if (!o) return;
      o.color = $('sel-color').value;
      this.syncOutlinerColors();
      this.invalidate();
    });
    $('sel-prop').addEventListener('input', () => {
      const o = this.selectedObject;
      if (!o) return;
      o.prop = Number($('sel-prop').value);
      $('sel-prop-value').textContent = o.prop.toFixed(2);
      this.invalidate();
    });
    $('sel-size-link').addEventListener('change', () => {
      this.sizeLink = $('sel-size-link').checked;
      this.syncInspector();
    });
    for (const axis of ['x', 'y', 'z']) {
      $(`sel-size-${axis}`).addEventListener('input', () => {
        const o = this.selectedObject;
        if (!o) return;
        const v = clamp(Number($(`sel-size-${axis}`).value), 0.1, 8);
        if (o.type === 1 || this.sizeLink) o.scale = [v, v, v];
        else o.scale[['x', 'y', 'z'].indexOf(axis)] = v;
        this.syncSizeFields(o);
        this.invalidate();
      });
    }
    $('btn-duplicate').addEventListener('click', () => this.duplicate());
    $('btn-delete').addEventListener('click', () => this.deleteSelected());
    $('btn-floor').addEventListener('click', () => this.dropToFloor());
    $('btn-focus').addEventListener('click', () => this.focusSelected());

    // world
    $('env-time').addEventListener('input', () => {
      this.env.time = Number($('env-time').value);
      $('time-label').textContent = formatClock(this.env.time);
      this.invalidate();
    });
    $('env-animate').addEventListener('change', () => {
      this.env.animate = $('env-animate').checked;
    });
    $('env-sun-color').addEventListener('input', () => {
      this.env.sunColor = $('env-sun-color').value;
      this.invalidate();
    });
    $('fx-exposure').addEventListener('input', () => {
      this.env.exposure = Number($('fx-exposure').value);
      $('exposure-value').textContent = this.env.exposure.toFixed(2);
      this.invalidate();
    });
    $('env-fog').addEventListener('input', () => {
      this.env.fog = Number($('env-fog').value);
      $('fog-value').textContent = this.env.fog.toFixed(3);
      this.invalidate();
    });
    $('env-aniso').addEventListener('input', () => {
      this.env.aniso = Number($('env-aniso').value);
      $('aniso-value').textContent = this.env.aniso.toFixed(2);
      this.invalidate();
    });
    $('env-fogheight').addEventListener('input', () => {
      this.env.fogHeight = Number($('env-fogheight').value);
      $('fogheight-value').textContent = this.env.fogHeight.toFixed(2);
      this.invalidate();
    });
    $('env-fogglow').addEventListener('input', () => {
      this.env.fogGlow = Number($('env-fogglow').value);
      $('fogglow-value').textContent = this.env.fogGlow.toFixed(2);
      this.invalidate();
    });
    $('env-vol').addEventListener('change', () => {
      this.env.volSteps = Number($('env-vol').value);
      this.invalidate();
    });
    $('env-floor-color').addEventListener('input', () => {
      this.env.floorColor = $('env-floor-color').value;
      this.invalidate();
    });
    $('env-floor-style').addEventListener('change', () => {
      this.env.floorStyle = Number($('env-floor-style').value);
      this.invalidate();
    });
    $('env-floor-gloss').addEventListener('input', () => {
      this.env.floorGloss = Number($('env-floor-gloss').value);
      $('gloss-value').textContent = this.env.floorGloss.toFixed(2);
      this.invalidate();
    });
    $('fx-bloom').addEventListener('input', () => {
      this.env.bloom = Number($('fx-bloom').value);
      $('bloom-value').textContent = this.env.bloom.toFixed(2);
      this.invalidate();
    });
    $('fx-aperture').addEventListener('input', () => {
      this.env.aperture = Number($('fx-aperture').value);
      $('aperture-value').textContent = this.env.aperture.toFixed(3);
      this.invalidate();
    });
    $('fx-focus-auto').addEventListener('change', () => {
      this.env.autoFocus = $('fx-focus-auto').checked;
      $('fx-focus').disabled = this.env.autoFocus;
      this.invalidate();
    });
    $('fx-focus').addEventListener('input', () => {
      this.env.focus = Number($('fx-focus').value);
      $('focus-value').textContent = this.env.focus.toFixed(1);
      this.invalidate();
    });

    // render / quality
    $('q-preset').addEventListener('change', () => {
      const v = $('q-preset').value;
      if (v !== 'custom') this.applyQuality(v);
    });
    $('q-scale').addEventListener('change', () => {
      const v = $('q-scale').value;
      if (v === 'auto') {
        this.settings.autoRes = true;
        this.engine.autoScale = this.settings.resScale || 0.75;
      } else {
        this.settings.autoRes = false;
        this.settings.resScale = Number(v);
      }
      this.markCustom();
      this.invalidate();
    });
    $('q-bounces').addEventListener('input', () => {
      this.settings.bounces = Number($('q-bounces').value);
      $('bounces-value').textContent = String(this.settings.bounces);
      this.markCustom();
      this.invalidate();
    });
    $('q-samples').addEventListener('change', () => {
      this.settings.maxSamples = Number($('q-samples').value);
      this.markCustom();
      this.invalidate();
    });
    $('q-fps').addEventListener('change', () => {
      this.settings.fpsLimit = Number($('q-fps').value);
      this.markCustom();
    });
    $('q-denoise').addEventListener('change', () => {
      this.settings.denoise = $('q-denoise').checked;
      this.markCustom();
      this.invalidate();
    });
    $('q-outlines').addEventListener('change', () => {
      this.settings.outlines = $('q-outlines').checked;
      this.markCustom();
      this.invalidate();
    });
    for (const [id, key] of [
      ['g-volumetrics', 'volumetrics'],
      ['g-bloom', 'bloomFx'],
      ['g-dof', 'dof'],
      ['g-shadows', 'shadows'],
      ['g-reflections', 'reflections'],
    ]) {
      $(id).addEventListener('change', () => {
        this.settings[key] = $(id).checked;
        // Depth of field with a zero aperture is invisible, so give it one.
        if (key === 'dof' && this.settings.dof && this.env.aperture < 0.004) {
          this.env.aperture = 0.08;
          this.syncEnvUI();
        }
        this.toast(`${GFX_LABELS[key]} ${this.settings[key] ? 'on' : 'off'}`);
        this.invalidate();
      });
    }

    $('cam-fov').addEventListener('input', () => {
      this.player.fov = Number($('cam-fov').value);
      this.player.updateVectors();
      $('fov-value').textContent = `${Math.round(this.player.fov)}°`;
      this.invalidate();
    });
    $('cam-speed').addEventListener('input', () => {
      this.moveSpeed = Number($('cam-speed').value);
      $('speed-value').textContent = this.moveSpeed.toFixed(1);
    });
    $('cam-sens').addEventListener('input', () => {
      this.sensitivity = Number($('cam-sens').value);
      $('sens-value').textContent = `${this.sensitivity.toFixed(2)}×`;
    });
    $('cam-visible').addEventListener('change', () => {
      this.player.visible = $('cam-visible').checked;
      this.invalidate();
    });
    $('cam-color').addEventListener('input', () => {
      this.player.color = $('cam-color').value;
      this.invalidate();
    });

    $('rcam-spawn').addEventListener('click', () => this.spawnRenderCam());
    $('rcam-relocate').addEventListener('click', () => this.relocateRenderCam());
    $('rcam-remove').addEventListener('click', () => this.removeRenderCam());
    $('rcam-clear').addEventListener('click', () => {
      this.fly = false;
      $('rcam-fly').checked = false;
      this.syncCameraUI();
      this.invalidate();
    });
    $('rcam-fly').addEventListener('change', () => {
      this.fly = $('rcam-fly').checked && !!this.renderCam;
      this.syncCameraUI();
      this.invalidate();
    });
    $('rcam-pip').addEventListener('change', () => {
      this.pipEnabled = $('rcam-pip').checked;
      this.engine.pipAccum = 0;
      this.invalidate();
    });
    $('rcam-visible').addEventListener('change', () => {
      if (this.renderCam) this.renderCam.visible = $('rcam-visible').checked;
      this.invalidate();
    });
    $('rcam-fov').addEventListener('input', () => {
      if (!this.renderCam) return;
      this.renderCam.fov = Number($('rcam-fov').value);
      this.renderCam.updateVectors();
      $('rfov-value').textContent = `${Math.round(this.renderCam.fov)}°`;
      this.invalidate();
    });
    $('rcam-color').addEventListener('input', () => {
      if (!this.renderCam) return;
      this.renderCam.color = $('rcam-color').value;
      this.invalidate();
    });

    $('btn-render').addEventListener('click', () => this.startFinalRender());
    $('btn-cancel-render').addEventListener('click', () => this.engine.cancelExport());
  }

  bindInput() {
    window.addEventListener('resize', () => {
      this.invalidate();
    });

    document.addEventListener('pointerlockchange', () => {
      this.locked = document.pointerLockElement === this.canvas;
      $('crosshair').hidden = !this.locked;
      $('btn-look').classList.toggle('on', this.locked);
      if (this.locked) this.setHot(0);
      this.updateHint();
    });

        this.canvas.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 || this.exporting) return;
      if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
      if (this.locked) {
        const ray = this.viewCam.ray(0, 0, window.innerWidth / window.innerHeight);
        const id = this.pick(ray);
        this.select(id);
        return;
      }
      const ray = this.rayFromEvent(e);
      const axis = this.hitGizmo(ray);
      if (axis > 0) {
        this.beginDrag(axis, e, ray);
        return;
      }
      this.select(this.pick(ray));
    });

    this.canvas.addEventListener('dblclick', () => {
      if (!this.locked && !this.exporting) this.toggleLook();
    });

    window.addEventListener('pointermove', (e) => {
      if (this.locked) {
        const cam = this.viewCam;
        cam.yaw += e.movementX * 0.0022 * this.sensitivity;
        cam.pitch = clamp(cam.pitch - e.movementY * 0.0022 * this.sensitivity, -1.5, 1.5);
        cam.updateVectors();
        this.invalidate();
      } else if (this.gizmo.dragging) {
        this.dragMove(e);
      } else {
        this.updateHover(e);
      }
    });

    window.addEventListener('pointerup', () => {
      if (this.gizmo.dragging) this.endDrag();
    });

    window.addEventListener('keydown', (e) => {
      const tag = (e.target && e.target.tagName) || '';
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || this.exporting) return;
      const k = e.key.toLowerCase();
      if (k === 'tab') {
        if (tag === 'BODY' || tag === 'CANVAS') {
          e.preventDefault();
          this.toggleLook();
        }
        return;
      }
      if (k === ' ' || k === 'arrowleft' || k === 'arrowright' || k === 'arrowup' || k === 'arrowdown') e.preventDefault();
      this.keys[k] = true;
      if (e.shiftKey) this.keys.shift = true;
      if (k === 'arrowright') this.select((this.selected + 1) % Math.max(1, this.objects.length));
      else if (k === 'arrowleft') this.select((this.selected - 1 + this.objects.length) % Math.max(1, this.objects.length));
      else if (k === 'delete' || k === 'backspace') this.deleteSelected();
      else if (k === 'd' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        this.duplicate();
      } else if (k === '1') this.spawn(0);
      else if (k === '2') this.spawn(1);
      else if (k === 'f') this.screenshot();
      else if (k === 'h') this.openModal('modal-help');
    });

    window.addEventListener('keyup', (e) => {
      const k = e.key.toLowerCase();
      this.keys[k] = false;
      if (!e.shiftKey) this.keys.shift = false;
    });

    window.addEventListener('blur', () => {
      this.keys = {};
    });
  }

  invalidate() {
    if (this.engine) this.engine.invalidate();
  }

  // ---------------------------------------------------------------- camera
  toggleLook() {
    if (document.pointerLockElement === this.canvas) {
      document.exitPointerLock();
    } else if (this.canvas.requestPointerLock) {
      const p = this.canvas.requestPointerLock();
      if (p && p.catch) p.catch(() => {});
    }
  }

  updateHint() {
    const t = $('hint-text');
    if (!t) return;
    if (this.exporting) t.textContent = 'Rendering — press Cancel to stop.';
    else if (this.locked) t.textContent = 'WASD move · Space/Ctrl rise & descend · Shift boost · Esc to edit with gizmos';
    else if (this.selectedObject) t.textContent = 'Drag the coloured gizmo arrows · centre cube moves freely · Shift snaps';
    else t.textContent = 'Click an object to select · double-click or Look to fly · 1/2 add shapes';
  }

  // ---------------------------------------------------------------- gizmo
  gizmoInfo() {
    const o = this.selectedObject;
    if (!o) return null;
    const cam = this.viewCam;
    const dist = len3(sub3(o.pos, cam.pos));
    return { pos: o.pos, scale: clamp(dist * 0.15, 0.45, 3.5), hot: this.gizmo.hot };
  }

  rayFromEvent(e) {
    return rayFromScreen(this.viewCam, e.clientX, e.clientY, window.innerWidth, window.innerHeight);
  }

  hitGizmo(ray) {
    const g = this.gizmoInfo();
    if (!g) return 0;
    let best = 0;
    let bestDist = Infinity;
    for (let i = 0; i < 3; i++) {
      const r = closestRayToAxis(ray, g.pos, AXES[i], g.scale * 1.32);
      if (r.dist < 0.125 * g.scale && r.dist < bestDist) {
        bestDist = r.dist;
        best = i + 1;
      }
    }
    const toC = sub3(g.pos, ray.origin);
    const t = dot3(toC, ray.dir);
    if (t > 0) {
      const closest = add3(ray.origin, mul3(ray.dir, t));
      if (len3(sub3(g.pos, closest)) < 0.15 * g.scale) best = 4;
    }
    return best;
  }

  setHot(axis) {
    if (this.gizmo.hot === axis) return;
    this.gizmo.hot = axis;
    this.canvas.classList.toggle('hot', axis > 0);
    this.invalidate();
  }

  updateHover(e) {
    if (!this.gizmoInfo()) {
      this.setHot(0);
      return;
    }
    this.setHot(this.hitGizmo(this.rayFromEvent(e)));
  }

  beginDrag(axis, e, ray) {
    const o = this.selectedObject;
    if (!o) return;
    this.gizmo.dragging = true;
    this.gizmo.axis = axis;
    this.gizmo.startPos = [...o.pos];
    this.gizmo.startU = 0;
    this.gizmo.startPoint = null;
    this.setHot(axis);
    this.canvas.classList.add('dragging');
    if (axis === 4) {
      this.gizmo.startPoint = rayPlane(ray, this.gizmo.startPos, this.viewCam.forward);
    } else {
      this.gizmo.startU = closestRayToAxis(ray, this.gizmo.startPos, AXES[axis - 1]).u;
    }
  }

  dragMove(e) {
    const o = this.selectedObject;
    if (!o) return;
    const ray = this.rayFromEvent(e);
    let np = null;
    if (this.gizmo.axis === 4) {
      const hit = rayPlane(ray, this.gizmo.startPos, this.viewCam.forward);
      if (!hit || !this.gizmo.startPoint) return;
      np = add3(this.gizmo.startPos, sub3(hit, this.gizmo.startPoint));
    } else {
      const axis = AXES[this.gizmo.axis - 1];
      const u = closestRayToAxis(ray, this.gizmo.startPos, axis).u;
      np = add3(this.gizmo.startPos, mul3(axis, u - this.gizmo.startU));
    }
    if (e.shiftKey) np = np.map((v) => Math.round(v / 0.25) * 0.25);
    o.pos = np;
    this.updatePoseReadout();
    this.invalidate();
  }

  endDrag() {
    this.gizmo.dragging = false;
    this.canvas.classList.remove('dragging');
    this.setHot(0);
    this.updateHint();
  }

  pick(ray) {
    let best = -1;
    let bestT = Infinity;
    for (let i = 0; i < this.objects.length; i++) {
      const t = rayObject(ray, this.objects[i]);
      if (t > 0 && t < bestT) {
        bestT = t;
        best = i;
      }
    }
    return best;
  }

  // ---------------------------------------------------------------- object ops
  spawn(type) {
    if (this.objects.length >= MAX_OBJECTS) {
      this.toast('Object limit reached (96)');
      return;
    }
    const cam = this.viewCam;
    const p = add3(cam.pos, mul3(cam.forward, 3.6));
    let size = 1.2;
    const palette = ['#ff6b9d', '#ffd479', '#67e8f9', '#a78bfa', '#86efac'];
    const mat = Math.floor(Math.random() * 4);
    const o = makeObject(type, [p[0], Math.max(size * 0.5, p[1]), p[2]], size, mat, palette[Math.floor(Math.random() * palette.length)]);
    o.prop = MATERIALS[mat].def;
    this.objects.push(o);
    this.select(this.objects.length - 1);
    this.rebuildOutliner();
    this.invalidate();
  }

  clearScene() {
    this.objects = [];
    this.select(-1);
    this.rebuildOutliner();
    this.invalidate();
    this.toast('Scene cleared');
  }

  remix() {
    remixObjects(this.objects, Date.now());
    this.syncInspector();
    this.rebuildOutliner();
    this.invalidate();
    this.toast('Materials remixed');
  }

  deleteSelected() {
    if (this.selected < 0) return;
    this.objects.splice(this.selected, 1);
    this.select(-1);
    this.rebuildOutliner();
    this.invalidate();
  }

  duplicate() {
    const o = this.selectedObject;
    if (!o) return;
    if (this.objects.length >= MAX_OBJECTS) {
      this.toast('Object limit reached (96)');
      return;
    }
    const copy = clampObject(JSON.parse(JSON.stringify(o)));
    copy.pos[0] += 0.9;
    this.objects.push(copy);
    this.select(this.objects.length - 1);
    this.rebuildOutliner();
    this.invalidate();
  }

  dropToFloor() {
    const o = this.selectedObject;
    if (!o) return;
    o.pos[1] = o.type === 1 ? o.scale[0] * 0.5 : o.scale[1] * 0.5;
    this.syncInspector();
    this.invalidate();
  }

  focusSelected() {
    const o = this.selectedObject;
    if (!o) return;
    this.env.autoFocus = false;
    this.env.focus = Math.max(0.3, len3(sub3(o.pos, this.viewCam.pos)));
    this.syncEnvUI();
    this.invalidate();
    this.toast('DOF focused on selection');
  }

  select(id) {
    this.selected = id >= 0 && id < this.objects.length ? id : -1;
    this.syncInspector();
    this.updateOutlinerSelection();
    this.updateHint();
    this.invalidate();
  }

  // ---------------------------------------------------------------- presets
  applyPreset(name, silent = false) {
    const preset = PRESETS[name];
    if (!preset) return;
    const data = preset.build();
    this.objects = data.objects.map(clampObject);
    this.env = { ...DEFAULT_ENV, ...data.env };
    this.selected = -1;
    this.renderCam = null;
    this.fly = false;
    this.pipEnabled = true;
    this.player.pos = [...data.camera.pos];
    this.player.setFov(data.camera.fov);
    this.player.lookAt(data.camera.target);
    this.currentPreset = name;
    $('scene-name').textContent = preset.label;
    document.querySelectorAll('.preset-btn').forEach((b) => b.classList.toggle('active', b.dataset.preset === name));
    this.syncAllUI();
    this.rebuildOutliner();
    this.invalidate();
    if (!silent) this.toast(`Loaded “${preset.label}”`);
  }

  applyQuality(name, silent = false) {
    const q = QUALITY_PRESETS[name];
    if (!q) return;
    this.settings.autoRes = q.autoRes;
    this.settings.resScale = q.scale;
    this.settings.bounces = q.bounces;
    this.settings.maxSamples = q.samples;
    this.settings.fpsLimit = q.fps;
    this.settings.denoise = q.denoise;
    this.env.volSteps = q.vol;
    this.env.bloom = q.bloom;
    this.engine.autoScale = q.scale;
    this.qualityPreset = name;
    this.syncQualityUI();
    this.syncEnvUI();
    this.invalidate();
    if (!silent) this.toast(`Quality: ${name}`);
  }

  markCustom() {
    this.qualityPreset = 'custom';
    if ($('q-preset').value !== 'custom') $('q-preset').value = 'custom';
  }

  // ---------------------------------------------------------------- render cam
  spawnRenderCam() {
    const cam = new Camera({
      pos: [...this.player.pos],
      yaw: this.player.yaw,
      pitch: this.player.pitch,
      fov: 50,
      color: '#2b3f8f',
    });
    this.renderCam = cam;
    this.pipEnabled = true;
    $('rcam-pip').checked = true;
    this.syncCameraUI();
    this.syncEnvUI();
    this.invalidate();
    this.toast('Render camera spawned');
  }

  relocateRenderCam() {
    if (!this.renderCam) return;
    const c = this.renderCam;
    c.pos = [...this.player.pos];
    c.yaw = this.player.yaw;
    c.pitch = this.player.pitch;
    c.updateVectors();
    this.invalidate();
    this.toast('Render camera relocated');
  }

  removeRenderCam() {
    this.renderCam = null;
    this.fly = false;
    this.syncCameraUI();
    this.invalidate();
    this.toast('Render camera removed');
  }

  // ---------------------------------------------------------------- lens
  applyLens(cam) {
    cam.aperture = this.env.aperture;
    if (this.env.autoFocus && this.selectedObject) {
      cam.focus = Math.max(0.3, len3(sub3(this.selectedObject.pos, cam.pos)));
    } else {
      cam.focus = this.env.focus;
    }
  }

  // ---------------------------------------------------------------- UI sync
  syncAllUI() {
    this.syncEnvUI();
    this.syncQualityUI();
    this.syncCameraUI();
    this.syncInspector();
    this.updateHint();
  }

  syncEnvUI() {
    this.uiSync = true;
    const e = this.env;
    $('env-time').value = e.time;
    $('time-label').textContent = formatClock(e.time);
    $('env-animate').checked = !!e.animate;
    $('env-sun-color').value = e.sunColor;
    $('fx-exposure').value = e.exposure;
    $('exposure-value').textContent = e.exposure.toFixed(2);
    $('env-fog').value = e.fog;
    $('fog-value').textContent = e.fog.toFixed(3);
    $('env-aniso').value = e.aniso;
    $('aniso-value').textContent = e.aniso.toFixed(2);
    $('env-fogheight').value = e.fogHeight;
    $('fogheight-value').textContent = e.fogHeight.toFixed(2);
    $('env-fogglow').value = e.fogGlow;
    $('fogglow-value').textContent = e.fogGlow.toFixed(2);
    $('env-vol').value = String(e.volSteps);
    $('env-floor-color').value = e.floorColor;
    $('env-floor-style').value = String(e.floorStyle);
    $('env-floor-gloss').value = e.floorGloss;
    $('gloss-value').textContent = e.floorGloss.toFixed(2);
    $('fx-bloom').value = e.bloom;
    $('bloom-value').textContent = e.bloom.toFixed(2);
    $('fx-aperture').value = e.aperture;
    $('aperture-value').textContent = e.aperture.toFixed(3);
    $('fx-focus-auto').checked = !!e.autoFocus;
    $('fx-focus').value = e.focus;
    $('focus-value').textContent = e.focus.toFixed(1);
    $('fx-focus').disabled = !!e.autoFocus;
    this.uiSync = false;
  }

  syncQualityUI() {
    this.uiSync = true;
    const s = this.settings;
    $('q-preset').value = this.qualityPreset;
    $('q-scale').value = s.autoRes ? 'auto' : String(s.resScale);
    $('q-bounces').value = s.bounces;
    $('bounces-value').textContent = String(s.bounces);
    $('q-samples').value = String(s.maxSamples);
    $('q-fps').value = String(s.fpsLimit);
    $('q-denoise').checked = !!s.denoise;
    $('q-outlines').checked = !!s.outlines;
    $('g-volumetrics').checked = s.volumetrics !== false;
    $('g-bloom').checked = s.bloomFx !== false;
    $('g-dof').checked = s.dof !== false;
    $('g-shadows').checked = s.shadows !== false;
    $('g-reflections').checked = s.reflections !== false;
    $('cam-fov').value = this.player.fov;
    $('fov-value').textContent = `${Math.round(this.player.fov)}°`;
    $('cam-speed').value = this.moveSpeed;
    $('speed-value').textContent = this.moveSpeed.toFixed(1);
    $('cam-sens').value = this.sensitivity;
    $('sens-value').textContent = `${this.sensitivity.toFixed(2)}×`;
    $('cam-visible').checked = this.player.visible;
    $('cam-color').value = this.player.color;
    this.uiSync = false;
  }

  syncCameraUI() {
    const rc = this.renderCam;
    $('rcam-relocate').disabled = !rc;
    $('rcam-remove').disabled = !rc;
    $('rcam-clear').disabled = !rc;
    $('rcam-fly').disabled = !rc;
    $('rcam-pip').disabled = !rc;
    $('rcam-visible').disabled = !rc;
    $('rcam-fov').disabled = !rc;
    $('rcam-color').disabled = !rc;
    $('rcam-fly').checked = this.fly && !!rc;
    $('rcam-pip').checked = this.pipEnabled && !!rc;
    if (rc) {
      $('rcam-fov').value = rc.fov;
      $('rfov-value').textContent = `${Math.round(rc.fov)}°`;
      $('rcam-color').value = rc.color;
      $('rcam-visible').checked = rc.visible;
    }
    const pipVisible = !!(rc && this.pipEnabled && !this.fly && !this.exporting);
    $('pip-frame').hidden = !pipVisible;
    document.body.classList.toggle('pip-on', pipVisible);
    this.updateHint();
    if (!pipVisible && this.engine) this.engine.pipAccum = 0;
  }

  syncInspector() {
    const o = this.selectedObject;
    $('sel-empty').hidden = !!o;
    $('sel-body').hidden = !o;
    if (!o) return;
    const meta = MATERIALS[o.mat];
    $('sel-name').textContent = o.type === 1 ? 'Sphere' : 'Cube';
    $('sel-id').textContent = `#${this.selected + 1}`;
    $('sel-dot').style.color = o.color;
    $('sel-dot').style.background = o.color;
    this.updatePoseReadout();
    $('sel-material').value = String(o.mat);
    $('sel-color').value = o.color;
    $('sel-prop-label').textContent = meta.prop;
    const prop = $('sel-prop');
    prop.min = meta.min;
    prop.max = meta.max;
    prop.step = meta.step;
    prop.value = o.prop;
    $('sel-prop-value').textContent = o.prop.toFixed(2);
    this.sizeLink = $('sel-size-link').checked;
    this.syncSizeFields(o);
    $('size-yz-row').hidden = o.type === 1;
    $('size-z-row').hidden = o.type === 1;
  }

  syncSizeFields(o) {
    $('sel-size-x').value = o.scale[0];
    $('sel-size-y').value = o.scale[1];
    $('sel-size-z').value = o.scale[2];
    $('size-x-val').textContent = o.scale[0].toFixed(1);
    $('size-y-val').textContent = o.scale[1].toFixed(1);
    $('size-z-val').textContent = o.scale[2].toFixed(1);
  }

  updatePoseReadout() {
    const o = this.selectedObject;
    if (!o) return;
    $('pos-x').textContent = o.pos[0].toFixed(2);
    $('pos-y').textContent = o.pos[1].toFixed(2);
    $('pos-z').textContent = o.pos[2].toFixed(2);
  }

  rebuildOutliner() {
    const host = $('outliner');
    host.innerHTML = '';
    if (!this.objects.length) {
      const d = document.createElement('div');
      d.className = 'out-empty';
      d.textContent = 'Empty scene. Add a cube or sphere.';
      host.appendChild(d);
    }
    this.objects.forEach((o, i) => {
      const item = document.createElement('div');
      item.className = 'out-item' + (i === this.selected ? ' active' : '');
      item.dataset.index = String(i);
      const dot = document.createElement('span');
      dot.className = 'dot';
      dot.style.color = o.color;
      dot.style.background = o.color;
      const nm = document.createElement('span');
      nm.className = 'nm';
      nm.textContent = objectLabel(o, i);
      const mt = document.createElement('span');
      mt.className = 'mt';
      mt.textContent = MATERIALS[o.mat].name;
      item.append(dot, nm, mt);
      item.addEventListener('click', () => this.select(i));
      host.appendChild(item);
    });
    $('outliner-count').textContent = `${this.objects.length}/${MAX_OBJECTS}`;
    $('stat-objs').textContent = `${this.objects.length}/${MAX_OBJECTS}`;
  }

  updateOutlinerSelection() {
    const host = $('outliner');
    [...host.children].forEach((el) => {
      const idx = Number(el.dataset.index);
      el.classList.toggle('active', idx === this.selected);
    });
    if (!host.children.length && this.objects.length) this.rebuildOutliner();
  }

  syncOutlinerColors() {
    const host = $('outliner');
    [...host.children].forEach((el) => {
      const idx = Number(el.dataset.index);
      const o = this.objects[idx];
      if (!o) return;
      const dot = el.querySelector('.dot');
      if (dot) {
        dot.style.color = o.color;
        dot.style.background = o.color;
      }
    });
  }

  // ---------------------------------------------------------------- modals / toast
  openModal(id) {
    const el = $(id);
    if (el) el.classList.add('open');
  }

  closeModal(id) {
    const el = $(id);
    if (el) el.classList.remove('open');
  }

  toast(text) {
    const host = $('toast');
    const el = document.createElement('div');
    el.className = 'toast-item';
    el.textContent = text;
    host.appendChild(el);
    setTimeout(() => el.remove(), 2600);
  }

  // ---------------------------------------------------------------- save / load
  saveScene() {
    try {
      const payload = makeScenePayload(this.objects, this.env, this.player, this.renderCam);
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      downloadBlob(blob, `rtd4-scene-${Date.now()}.json`);
      this.toast('Scene saved');
    } catch (err) {
      console.error(err);
      this.toast('Could not save scene');
    }
  }

  loadSceneFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = parseScenePayload(JSON.parse(String(reader.result)));
        this.objects = parsed.objects;
        this.env = parsed.env;
        this.selected = -1;
        this.player = Camera.deserialize(parsed.camera || {});
        this.player.updateVectors();
        this.renderCam = parsed.renderCamera ? Camera.deserialize(parsed.renderCamera) : null;
        if (this.renderCam) this.renderCam.updateVectors();
        this.fly = false;
        this.qualityPreset = 'custom';
        document.querySelectorAll('.preset-btn').forEach((b) => b.classList.remove('active'));
        $('scene-name').textContent = file.name.replace(/\.json$/i, '');
        this.syncAllUI();
        this.rebuildOutliner();
        this.invalidate();
        this.toast('Scene loaded');
      } catch (err) {
        console.error(err);
        this.toast('Invalid scene file');
      }
      $('file-load').value = '';
    };
    reader.readAsText(file);
  }

  screenshot() {
    if (this.exporting) return;
    this.engine.requestScreenshot();
  }

  // ---------------------------------------------------------------- final render
  startFinalRender() {
    if (this.exporting) return;
    const samples = Number($('ex-samples').value);
    const [width, height] = $('ex-res').value.split('x').map(Number);
    const format = $('ex-format').value;
    const denoise = $('ex-denoise').value;
    this.applyLens(this.player);
    if (this.renderCam) this.applyLens(this.renderCam);
    const state = this.buildFrameState();
    state.exportCam = cloneCamera(this.renderCam || this.player);
    state.exportCam2 = this.renderCam ? cloneCamera(this.player) : null;
    state.exportActiveCam = this.renderCam ? 1 : 0;
    state.gizmo = null;
    state.pip = null;
    try {
      const ok = this.engine.startExport(state, { width, height, samples, format, denoise });
      if (!ok) return;
    } catch (err) {
      console.error(err);
      this.toast('Render failed to start');
      return;
    }
    this.exporting = true;
    this.engine.onExportProgress = (p) => this.onExportProgress(p);
    $('render-overlay').classList.add('open');
    const dn = denoise === 'none' ? 'no denoiser' : `${denoise} denoiser`;
    $('ro-phase').textContent = `${width}×${height} · ${samples} samples per tile · ${dn}`;
    $('ro-fill').style.width = '0%';
    $('ro-status').textContent = 'Preparing…';
    $('ro-eta').textContent = 'ETA —';
    $('pip-frame').hidden = true;
    this.updateHint();
  }

  onExportProgress(p) {
    $('ro-fill').style.width = `${(p.done / p.total) * 100}%`;
    $('ro-status').textContent = `Tile ${p.tileIndex + 1}/${p.tileCount} · sample ${p.sample}/${p.samples}`;
    $('ro-eta').textContent = `ETA ${formatDuration(p.etaMs)}`;
  }

  onExportDone(res) {
    this.exporting = false;
    $('render-overlay').classList.remove('open');
    this.syncCameraUI();
    this.invalidate();
    this.updateHint();
    if (res && res.cancelled) this.toast('Render cancelled');
    else if (res && res.filename) this.toast(`Saved ${res.filename}`);
    else if (res && res.error) this.toast('Export failed — try PNG or a smaller resolution');
  }

  // ---------------------------------------------------------------- state
  buildFrameState() {
    this.applyLens(this.player);
    if (this.renderCam) this.applyLens(this.renderCam);
    const viewIsRender = !!(this.fly && this.renderCam);
    const cam = viewIsRender ? this.renderCam : this.player;
    const other = viewIsRender ? this.player : this.renderCam;
    const g = this.gizmoInfo();
    return {
      objects: this.objects,
      settings: this.settings,
      env: this.env,
      envUniforms: computeEnvUniforms(this.env),
      cam,
      cam2: other || null,
      activeCam: viewIsRender ? 1 : 0,
      selected: this.selected,
      outlines: this.settings.outlines,
      gizmo: g,
      pip:
        this.renderCam && this.pipEnabled && !viewIsRender
          ? { cam: this.renderCam, cam2: this.player, activeCam: 1 }
          : null,
    };
  }

  // ---------------------------------------------------------------- loop
  loop(now) {
    requestAnimationFrame(this.loop.bind(this));
    const dt = Math.min(0.08, (now - this.lastFrame) / 1000);
    this.lastFrame = now;

    if (!this.exporting && this.env.animate) {
      this.env.time = (this.env.time + dt * 0.32) % 24;
      $('env-time').value = this.env.time;
      $('time-label').textContent = formatClock(this.env.time);
      this.invalidate();
    }

    if (!this.exporting && this.engine) {
      const tag = document.activeElement ? document.activeElement.tagName : '';
      if (tag !== 'INPUT' && tag !== 'SELECT') {
        const k = this.keys;
        const cam = this.viewCam;
        const f = (k.w ? 1 : 0) - (k.s ? 1 : 0);
        const s = (k.d ? 1 : 0) - (k.a ? 1 : 0);
        const lift = (k[' '] ? 1 : 0) - (k.control ? 1 : 0);
        if (f || s || lift) {
          const speed = this.moveSpeed * (k.shift ? 2.5 : 1);
          cam.move(f, s, lift, speed, dt);
          cam.updateVectors();
          this.invalidate();
        }
      }
    }

    if (this.engine && !this.engine.failed) {
      const state = this.buildFrameState();
      const drew = this.engine.render(state, now);
      if (drew) {
        this.drawTimes.push(now);
        if (this.drawTimes.length > 40) this.drawTimes.shift();
        const fps = this.drawTimes.length > 2 ? (1000 * (this.drawTimes.length - 1)) / (now - this.drawTimes[0]) : 0;
        this.fps = fps;
      }
    }

    if (now - this.lastHud > 140) {
      this.lastHud = now;
      this.updateHud(now);
    }
  }

  updateHud(now) {
    if (!this.engine) return;
    const e = this.engine;
    const fps = this.fps || 0;
    $('stat-fps').textContent = fps > 0 ? String(Math.round(fps)) : '—';
    $('stat-ms').textContent = fps > 0 ? `${(1000 / fps).toFixed(1)} ms` : '—';
    $('stat-spp').textContent = this.exporting ? 'render' : formatCount(e.accum);
    $('stat-res').textContent = `${e.width}×${e.height}`;
    const scalePct = e.width ? Math.round((e.width / Math.max(1, window.innerWidth)) * 100) : 0;
    $('stat-scale').textContent = `${scalePct}%`;
    $('hud-spp').textContent = this.exporting ? '—' : formatCount(e.accum);

    const maxed = this.settings.maxSamples > 0 && e.accum >= this.settings.maxSamples;
    const stateText = this.exporting ? 'RENDERING' : maxed ? 'CONVERGED' : 'SAMPLING';
    $('stat-state').textContent = stateText;
    $('stat-state').classList.toggle('converged', maxed && !this.exporting);
    $('hud-note').textContent = this.exporting
      ? 'Final render in progress'
      : maxed
        ? 'Converged — move the camera to resume sampling.'
        : 'Progressive accumulation — hold still to converge.';

    this.drawSparkline();

    const pipVisible = !!(this.renderCam && this.pipEnabled && !this.fly && !this.exporting);
    $('pip-frame').hidden = !pipVisible;
    document.body.classList.toggle('pip-on', pipVisible);
    this.updateHint();
    this.updatePoseReadout();
  }

  drawSparkline() {
    const canvas = $('sparkline');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    const n = this.drawTimes.length;
    if (n < 2) return;
    const intervals = [];
    for (let i = 1; i < n; i++) intervals.push(this.drawTimes[i] - this.drawTimes[i - 1]);
    const max = Math.max(20, ...intervals);
    ctx.beginPath();
    intervals.forEach((ms, i) => {
      const x = (i / Math.max(1, intervals.length - 1)) * w;
      const y = h - clamp(ms / max, 0, 1) * (h - 3) - 1;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    const grad = ctx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, 'rgba(79,214,255,0.35)');
    grad.addColorStop(1, 'rgba(79,214,255,0.95)');
    ctx.strokeStyle = grad;
    ctx.lineWidth = 1.4;
    ctx.stroke();
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    const fill = ctx.createLinearGradient(0, 0, 0, h);
    fill.addColorStop(0, 'rgba(79,214,255,0.18)');
    fill.addColorStop(1, 'rgba(79,214,255,0)');
    ctx.fillStyle = fill;
    ctx.fill();
  }
}

let app;
try {
  app = new App();
  window.app = app;
} catch (err) {
  console.error(err);
  const el = document.getElementById('error-screen');
  if (el) {
    document.getElementById('error-detail').textContent = err && err.message ? err.message : String(err);
    el.classList.add('open');
  }
}

export default app;
