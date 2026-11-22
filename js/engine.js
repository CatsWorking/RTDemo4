import { pathTraceFS, accumulateFS, prefilterFS, blurFS, displayFS, denoiseFS, quadVS } from './glsl.js';
import { hexToRgb, clamp, downloadBlob } from './utils.js';
import { MAX_OBJECTS, MAX_LIGHTS } from './scene.js';

const TEXELS_PER_OBJECT = 8;

// Final-render denoiser presets. Each preset is a chain of edge-aware a-trous
// passes with growing pixel steps: 1, 2, 4, 8. Extra passes widen the filter
// footprint without turning into a plain blur.
export const DENOISE_MODES = {
  none: { passes: 0, steps: [], sigma: 0.12, chroma: 0.12, strength: 1, pad: 0 },
  fast: { passes: 2, steps: [1, 2], sigma: 0.11, chroma: 0.11, strength: 1, pad: 8 },
  balanced: { passes: 3, steps: [1, 2, 4], sigma: 0.13, chroma: 0.13, strength: 1, pad: 16 },
  strong: { passes: 4, steps: [1, 2, 4, 8], sigma: 0.22, chroma: 0.2, strength: 1, pad: 32 },
};

function denoiseMode(name) {
  return DENOISE_MODES[name] || DENOISE_MODES.none;
}

class GLProgram {
  constructor(gl, label, fsSource) {
    this.gl = gl;
    this.label = label;
    this.prog = gl.createProgram();
    const vsh = this._shader(gl.VERTEX_SHADER, quadVS);
    const fsh = this._shader(gl.FRAGMENT_SHADER, fsSource);
    gl.attachShader(this.prog, vsh);
    gl.attachShader(this.prog, fsh);
    gl.linkProgram(this.prog);
    if (!gl.getProgramParameter(this.prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(this.prog);
      throw new Error(`${label} link failed:\n${log}`);
    }
    gl.deleteShader(vsh);
    gl.deleteShader(fsh);
    this.locations = new Map();
  }

  _shader(type, src) {
    const gl = this.gl;
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(s);
      const numbered = src
        .split('\n')
        .map((l, i) => `${String(i + 1).padStart(3)} | ${l}`)
        .join('\n');
      console.error(`Shader compile failed (${this.label}):\n${log}\n${numbered}`);
      throw new Error(`${this.label} compile failed: ${log}`);
    }
    return s;
  }

  loc(name) {
    if (!this.locations.has(name)) {
      this.locations.set(name, this.gl.getUniformLocation(this.prog, name));
    }
    return this.locations.get(name);
  }
}

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl2', {
      antialias: false,
      alpha: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: false,
      powerPreference: 'high-performance',
    });
    this.failed = false;
    this.failReason = '';
    if (!this.gl) {
      this.failed = true;
      this.failReason = 'WebGL2 is not available in this browser.';
      return;
    }
    const gl = this.gl;
    this.extFloat = gl.getExtension('EXT_color_buffer_float');
    if (!this.extFloat) {
      this.failed = true;
      this.failReason = 'EXT_color_buffer_float is unavailable, so HDR accumulation targets cannot be created.';
      return;
    }
    this.maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    this.width = 0;
    this.height = 0;
    this.accum = 0;
    this.ping = 0;
    this.autoScale = 0.75;
    this.gpuAvg = 0;
    this.measureTick = 0;
    this.lastWasConverged = false;
    this.shotRequested = false;
    this.exportJob = null;
    this.onExportProgress = null;
    this.onExportDone = null;
    this.onScreenshotReady = null;
    this.width = 0;
    this.lastDrawTime = 0;
    this._build();
  }

  _build() {
    const gl = this.gl;
    this.quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

    this.progTrace = new GLProgram(gl, 'path-trace', pathTraceFS);
    this.progAccum = new GLProgram(gl, 'accumulate', accumulateFS);
    this.progPrefilter = new GLProgram(gl, 'bloom-prefilter', prefilterFS);
    this.progBlur = new GLProgram(gl, 'bloom-blur', blurFS);
    this.progDisplay = new GLProgram(gl, 'display', displayFS);
    this.progDenoise = new GLProgram(gl, 'denoise', denoiseFS);

    this.objTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.objTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, TEXELS_PER_OBJECT, MAX_OBJECTS, 0, gl.RGBA, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.blackTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.blackTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    this.objData = new Float32Array(MAX_OBJECTS * TEXELS_PER_OBJECT * 4);
    this.lightIdx = new Int32Array(MAX_LIGHTS);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.disable(gl.CULL_FACE);
  }

  _target(w, h, float = true) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    if (float) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, null);
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    }
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`Framebuffer incomplete at ${w}x${h} (status ${status}).`);
    }
    return { tex, fbo, w, h };
  }

  _freeTarget(t) {
    if (!t) return;
    this.gl.deleteFramebuffer(t.fbo);
    this.gl.deleteTexture(t.tex);
  }

  ensureMain(w, h) {
    if (this.width === w && this.height === h && this.sampleTarget) return;
    const gl = this.gl;
    this._freeTarget(this.sampleTarget);
    this._freeTarget(this.accTargets && this.accTargets[0]);
    this._freeTarget(this.accTargets && this.accTargets[1]);
    this.sampleTarget = this._target(w, h, true);
    this.accTargets = [this._target(w, h, true), this._target(w, h, true)];
    this.width = w;
    this.height = h;
    this.accum = 0;
    this.ping = 0;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  ensurePip(w, h) {
    if (this.pipW === w && this.pipH === h && this.pipSample) return;
    this._freeTarget(this.pipSample);
    this._freeTarget(this.pipAcc && this.pipAcc[0]);
    this._freeTarget(this.pipAcc && this.pipAcc[1]);
    this.pipSample = this._target(w, h, true);
    this.pipAcc = [this._target(w, h, true), this._target(w, h, true)];
    this.pipW = w;
    this.pipH = h;
    this.pipAccum = 0;
    this.pipPing = 0;
  }

  ensureBloom(w, h) {
    if (this.bloomW === w && this.bloomH === h && this.bloomA) return;
    this._freeTarget(this.bloomA);
    this._freeTarget(this.bloomB);
    this.bloomA = this._target(w, h, true);
    this.bloomB = this._target(w, h, true);
    this.bloomW = w;
    this.bloomH = h;
  }

  ensureExport(w, h) {
    if (this.exW === w && this.exH === h && this.exSample) return;
    this._freeTarget(this.exSample);
    this._freeTarget(this.exAcc && this.exAcc[0]);
    this._freeTarget(this.exAcc && this.exAcc[1]);
    this._freeTarget(this.exDenoise && this.exDenoise[0]);
    this._freeTarget(this.exDenoise && this.exDenoise[1]);
    this._freeTarget(this.exDisplay);
    this.exSample = this._target(w, h, true);
    this.exAcc = [this._target(w, h, true), this._target(w, h, true)];
    this.exDenoise = [this._target(w, h, true), this._target(w, h, true)];
    this.exDisplay = this._target(w, h, false);
    this.exW = w;
    this.exH = h;
  }

  invalidate() {
    this.accum = 0;
    this.ping = 0;
    this.pipAccum = 0;
    this.pipPing = 0;
  }

  _deferReset() {
    this.resetNextFrame = true;
  }

  uploadObjects(objects) {
    const gl = this.gl;
    const data = this.objData;
    data.fill(0);
    let lights = 0;
    for (let i = 0; i < objects.length && i < MAX_OBJECTS; i++) {
      const o = objects[i];
      const base = i * TEXELS_PER_OBJECT * 4;
      const c = hexToRgb(o.color);
      data[base + 0] = o.pos[0];
      data[base + 1] = o.pos[1];
      data[base + 2] = o.pos[2];
      data[base + 3] = o.type;
      data[base + 4] = o.scale[0];
      data[base + 5] = o.scale[1];
      data[base + 6] = o.scale[2];
      data[base + 7] = o.mat;
      data[base + 8] = c[0];
      data[base + 9] = c[1];
      data[base + 10] = c[2];
      data[base + 11] = o.prop;
      if (o.mat === 3 && o.prop > 0 && lights < MAX_LIGHTS) {
        this.lightIdx[lights++] = i;
      }
    }
    this.lightCount = lights;
    gl.bindTexture(gl.TEXTURE_2D, this.objTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TEXELS_PER_OBJECT, MAX_OBJECTS, gl.RGBA, gl.FLOAT, data);
  }

  _bindQuad(prog) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    const loc = gl.getAttribLocation(prog.prog, 'a_position');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  _setCamera(prog, cam, activeCam) {
    const gl = this.gl;
    gl.uniform1i(prog.loc('u_activeCam'), activeCam);
    if (!cam) return;
    gl.uniform3fv(prog.loc('u_camPos'), cam.pos);
    gl.uniform3fv(prog.loc('u_camFwd'), cam.forward);
    gl.uniform3fv(prog.loc('u_camRight'), cam.right);
    gl.uniform3fv(prog.loc('u_camUp'), cam.up);
    gl.uniform3fv(prog.loc('u_camColor'), hexToRgb(cam.color));
    gl.uniform1f(prog.loc('u_camFov'), cam.fovScale);
    gl.uniform1f(prog.loc('u_camAperture'), cam.aperture || 0);
    gl.uniform1f(prog.loc('u_camFocus'), cam.focus || 6);
    gl.uniform1f(prog.loc('u_camVisible'), cam.visible ? 1 : 0);
  }

  _setCamera2(prog, cam) {
    const gl = this.gl;
    gl.uniform1i(prog.loc('u_cam2Active'), cam ? 1 : 0);
    if (!cam) return;
    gl.uniform3fv(prog.loc('u_cam2Pos'), cam.pos);
    gl.uniform3fv(prog.loc('u_cam2Fwd'), cam.forward);
    gl.uniform3fv(prog.loc('u_cam2Right'), cam.right);
    gl.uniform3fv(prog.loc('u_cam2Up'), cam.up);
    gl.uniform3fv(prog.loc('u_cam2Color'), hexToRgb(cam.color));
    gl.uniform1f(prog.loc('u_cam2Visible'), cam.visible ? 1 : 0);
  }

  tracePass(state, opts) {
    const gl = this.gl;
    const p = this.progTrace;
    gl.useProgram(p.prog);
    gl.bindFramebuffer(gl.FRAMEBUFFER, opts.target.fbo);
    gl.viewport(0, 0, opts.w, opts.h);

    gl.uniform2f(p.loc('u_resolution'), opts.w, opts.h);
    gl.uniform2f(p.loc('u_fullResolution'), opts.fullW, opts.fullH);
    gl.uniform2f(p.loc('u_tileOffset'), opts.tileX || 0, opts.tileY || 0);
    gl.uniform1i(p.loc('u_frame'), opts.frame);
    const gfx = state.settings;
    gl.uniform1i(p.loc('u_bounces'), state.settings.bounces);
    gl.uniform1i(p.loc('u_volSteps'), gfx.volumetrics === false ? 0 : state.env.volSteps);
    gl.uniform1i(p.loc('u_stars'), gfx.stars === false ? 0 : 1);

    const e = state.envUniforms;
    gl.uniform1f(p.loc('u_fogDensity'), gfx.volumetrics === false ? 0 : e.fogDensity);
    gl.uniform1f(p.loc('u_fogHeight'), e.fogHeight);
    gl.uniform1f(p.loc('u_fogAniso'), e.fogAniso);
    gl.uniform1f(p.loc('u_fogGlow'), e.fogGlow);
    gl.uniform1f(p.loc('u_sunRadius'), 0.022);
    gl.uniform3fv(p.loc('u_sunDir'), e.sunDir);
    gl.uniform3fv(p.loc('u_sunColor'), e.sunColor);
    gl.uniform3fv(p.loc('u_skyColor'), e.skyColor);
    gl.uniform3fv(p.loc('u_horizonColor'), e.horizonColor);
    gl.uniform3fv(p.loc('u_floorColor'), e.floorColor);
    gl.uniform1i(p.loc('u_floorStyle'), state.env.floorStyle);
    gl.uniform1f(p.loc('u_floorGloss'), gfx.reflections === false ? 0 : state.env.floorGloss);
    gl.uniform1i(p.loc('u_objCount'), Math.min(state.objects.length, MAX_OBJECTS));
    gl.uniform1i(p.loc('u_lightCount'), this.lightCount || 0);
    gl.uniform1iv(p.loc('u_lights[0]'), this.lightIdx);
    gl.uniform1i(p.loc('u_selected'), opts.selected ?? -1);
    gl.uniform1i(p.loc('u_showOutlines'), opts.outlines ? 1 : 0);

    const gz = opts.gizmo;
    gl.uniform1i(p.loc('u_gizmoMode'), gz ? 1 : 0);
    gl.uniform1i(p.loc('u_gizmoHot'), gz ? gz.hot : 0);
    gl.uniform1f(p.loc('u_gizmoScale'), gz ? gz.scale : 1);
    if (gz) gl.uniform3fv(p.loc('u_gizmoPos'), gz.pos);
    else gl.uniform3f(p.loc('u_gizmoPos'), 0, 0, 0);

    this._setCamera(p, opts.cam, opts.activeCam ?? 0);
    if (gfx.dof === false) gl.uniform1f(p.loc('u_camAperture'), 0);
    this._setCamera2(p, opts.cam2 || null);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.objTex);
    gl.uniform1i(p.loc('u_objTex'), 0);

    this._bindQuad(p);
  }

  accumulatePass(sampleTex, historyTex, targetFBO, w, h, frame) {
    const gl = this.gl;
    const p = this.progAccum;
    gl.useProgram(p.prog);
    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFBO);
    gl.viewport(0, 0, w, h);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sampleTex);
    gl.uniform1i(p.loc('u_current'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, historyTex);
    gl.uniform1i(p.loc('u_history'), 1);
    gl.uniform1f(p.loc('u_weight'), 1 / (frame + 1));
    this._bindQuad(p);
  }

  bloomPass(accumTex, w, h) {
    const gl = this.gl;
    const bw = Math.max(1, Math.floor(w / 4));
    const bh = Math.max(1, Math.floor(h / 4));
    this.ensureBloom(bw, bh);

    const pf = this.progPrefilter;
    gl.useProgram(pf.prog);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomA.fbo);
    gl.viewport(0, 0, bw, bh);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, accumTex);
    gl.uniform1i(pf.loc('u_tex'), 0);
    gl.uniform1f(pf.loc('u_threshold'), 0.85);
    this._bindQuad(pf);

    const bl = this.progBlur;
    gl.useProgram(bl.prog);
    gl.uniform1i(bl.loc('u_tex'), 0);
    const pass = (src, dst, dx, dy) => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fbo);
      gl.viewport(0, 0, bw, bh);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, src.tex);
      gl.uniform2f(bl.loc('u_dir'), dx / bw, dy / bh);
      this._bindQuad(bl);
    };
    pass(this.bloomA, this.bloomB, 1.3, 0);
    pass(this.bloomB, this.bloomA, 0, 1.3);
    pass(this.bloomA, this.bloomB, 2.8, 0);
    pass(this.bloomB, this.bloomA, 0, 2.8);
    return this.bloomA.tex;
  }

  displayPass(target, vx, vy, vw, vh, tex, bloomTex, opts) {
    const gl = this.gl;
    const p = this.progDisplay;
    gl.useProgram(p.prog);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.fbo : null);
    gl.viewport(vx, vy, vw, vh);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(p.loc('u_tex'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, bloomTex || this.blackTex);
    gl.uniform1i(p.loc('u_bloom'), 1);
    gl.uniform2f(p.loc('u_texel'), 1 / (opts.srcW || vw), 1 / (opts.srcH || vh));
    gl.uniform1f(p.loc('u_exposure'), opts.exposure ?? 1);
    gl.uniform1f(p.loc('u_bloomStrength'), opts.bloomStrength ?? 0);
    gl.uniform1f(p.loc('u_vignette'), opts.vignette ?? 0.35);
    gl.uniform1f(p.loc('u_denoise'), opts.denoise ? 1 : 0);
    const origin = opts.tileOrigin;
    const size = opts.imageSize;
    gl.uniform2f(p.loc('u_tileOrigin'), origin ? origin[0] : 0, origin ? origin[1] : 0);
    gl.uniform2f(p.loc('u_imageSize'), size ? size[0] : 0, size ? size[1] : 0);
    this._bindQuad(p);
  }

  getScale(state) {
    if (state.settings.autoRes) return clamp(this.autoScale, 0.3, 1.0);
    return clamp(state.settings.resScale, 0.25, 2.0);
  }

  _adaptCost(ms, limit) {
    if (!Number.isFinite(ms) || ms <= 0) return;
    this.gpuAvg = this.gpuAvg > 0 ? this.gpuAvg * 0.55 + ms * 0.45 : ms;
    const target = limit > 0 ? 1000 / limit : 1000 / 60;
    let changed = false;
    if (this.gpuAvg > target * 1.2 && this.autoScale > 0.3) {
      this.autoScale = Math.max(0.3, this.autoScale * 0.92);
      changed = true;
    } else if (this.gpuAvg < target * 0.55 && this.autoScale < 1.0) {
      this.autoScale = Math.min(1.0, this.autoScale * 1.04);
      changed = true;
    }
    if (changed) this._deferReset();
  }

  render(state, now) {
    if (this.failed) return false;
    const limit = state.settings.fpsLimit;
    if (limit > 0 && this.lastDrawTime && now - this.lastDrawTime < 1000 / limit - 1) {
      return false;
    }
    this.lastDrawTime = now;
    if (this.resetNextFrame) {
      this.resetNextFrame = false;
      this.invalidate();
    }

    const winW = Math.max(320, Math.floor(window.innerWidth));
    const winH = Math.max(240, Math.floor(window.innerHeight));
    if (this.canvas.width !== winW || this.canvas.height !== winH) {
      this.canvas.width = winW;
      this.canvas.height = winH;
    }

    if (this.exportJob) {
      this._stepExport(state);
      return true;
    }

    const gl = this.gl;
    const scale = this.getScale(state);
    const rw = Math.max(64, Math.floor(winW * scale));
    const rh = Math.max(64, Math.floor(winH * scale));
    this.ensureMain(rw, rh);
    this.uploadObjects(state.objects);

    const maxSamples = state.settings.maxSamples;
    const converged = maxSamples > 0 && this.accum >= maxSamples;
    this.lastWasConverged = converged;

    if (!converged) {
      const measure = !!(state.settings.autoRes && this.accum > 20 && this.measureTick++ % 20 === 0);
      this.tracePass(state, {
        target: this.sampleTarget,
        w: rw,
        h: rh,
        fullW: rw,
        fullH: rh,
        frame: this.accum,
        cam: state.cam,
        cam2: state.cam2,
        activeCam: state.activeCam,
        gizmo: state.gizmo,
        selected: state.selected,
        outlines: state.outlines,
      });
      const write = 1 - this.ping;
      this.accumulatePass(this.sampleTarget.tex, this.accTargets[this.ping].tex, this.accTargets[write].fbo, rw, rh, this.accum);
      this.ping = write;
      this.accum++;
      if (measure) {
        const px = new Float32Array(4);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.accTargets[this.ping].fbo);
        const t0 = performance.now();
        gl.readPixels(Math.floor(rw / 2), Math.floor(rh / 2), 1, 1, gl.RGBA, gl.FLOAT, px);
        this._adaptCost(performance.now() - t0, limit);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      }
    }

    const accumTex = this.accTargets[this.ping].tex;
    let bloomTex = null;
    if (state.settings.bloomFx !== false && state.env.bloom > 0.001) bloomTex = this.bloomPass(accumTex, rw, rh);
    const denoise = !!state.settings.denoise && this.accum < 160;
    this.displayPass(null, 0, 0, winW, winH, accumTex, bloomTex, {
      srcW: rw,
      srcH: rh,
      exposure: state.env.exposure,
      bloomStrength: state.env.bloom,
      vignette: 0.35,
      denoise,
    });

    if (state.pip) {
      this.ensurePip(360, 203);
      if (this.pipAccum < 1024) {
        this.tracePass(state, {
        target: this.pipSample,
        w: 360,
        h: 203,
        fullW: 360,
        fullH: 203,
        frame: this.pipAccum,
        cam: state.pip.cam,
        cam2: state.pip.cam2,
        activeCam: state.pip.activeCam,
          gizmo: null,
          selected: -1,
          outlines: false,
        });
        const write = 1 - this.pipPing;
        this.accumulatePass(this.pipSample.tex, this.pipAcc[this.pipPing].tex, this.pipAcc[write].fbo, 360, 203, this.pipAccum);
        this.pipPing = write;
        this.pipAccum = Math.min(this.pipAccum + 1, 1024);
      }
      this.displayPass(null, 22, 22, 360, 203, this.pipAcc[this.pipPing].tex, null, {
        srcW: 360,
        srcH: 203,
        exposure: state.env.exposure,
        bloomStrength: 0,
        vignette: 0.2,
        denoise: this.pipAccum < 80,
      });
    }

    if (this.shotRequested) {
      this.shotRequested = false;
      this._captureScreenshot();
    }
    return true;
  }

  _captureScreenshot() {
    const gl = this.gl;
    const w = this.canvas.width;
    const h = this.canvas.height;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const px = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    const cvs = document.createElement('canvas');
    cvs.width = w;
    cvs.height = h;
    const ctx = cvs.getContext('2d');
    const img = ctx.createImageData(w, h);
    const row = w * 4;
    for (let y = 0; y < h; y++) {
      img.data.set(px.subarray((h - 1 - y) * row, (h - y) * row), y * row);
    }
    ctx.putImageData(img, 0, 0);
    cvs.toBlob((blob) => {
      if (blob) downloadBlob(blob, `RTD4_shot_${w}x${h}.png`);
      if (this.onScreenshotReady) this.onScreenshotReady();
    }, 'image/png');
  }

  requestScreenshot() {
    this.shotRequested = true;
  }

  // ---------------- final render (tiled) ----------------
  startExport(state, opts) {
    if (this.exportJob) return false;
    const { width, height, samples, format } = opts;
    const mode = denoiseMode(opts.denoise);
    const tileSize = 1024;
    const tiles = [];
    for (let y = 0; y < height; y += tileSize) {
      for (let x = 0; x < width; x += tileSize) {
        tiles.push({
          x,
          y,
          w: Math.min(tileSize, width - x),
          h: Math.min(tileSize, height - y),
        });
      }
    }
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, width, height);

    this.uploadObjects(state.objects);

    this.exportJob = {
      width,
      height,
      samples,
      format,
      denoise: mode,
      pad: mode.pad,
      tiles,
      index: 0,
      sample: 0,
      doneSamples: 0,
      totalSamples: tiles.length * samples,
      started: performance.now(),
      canvas,
      ctx,
      tileCanvas: document.createElement('canvas'),
      cam: state.exportCam,
      cam2: state.exportCam2 || null,
      activeCam: state.exportActiveCam || 0,
      exposure: state.env.exposure,
      ping: 0,
      lastUi: 0,
      finishing: false,
    };
    const first = tiles[0];
    this.ensureExport(first.w + mode.pad * 2, first.h + mode.pad * 2);
    this.invalidate();
    return true;
  }

  cancelExport() {
    if (!this.exportJob || this.exportJob.finishing) return;
    this.exportJob = null;
    this.invalidate();
    if (this.onExportDone) this.onExportDone({ cancelled: true });
  }

  _stepExport(state) {
    const gl = this.gl;
    const job = this.exportJob;
    if (!job || job.finishing) return;
    const tile = job.tiles[job.index];
    const pad = job.pad;
    const pw = tile.w + pad * 2;
    const ph = tile.h + pad * 2;
    this.ensureExport(pw, ph);

    this.tracePass(state, {
      target: this.exSample,
      w: pw,
      h: ph,
      fullW: job.width,
      fullH: job.height,
      tileX: tile.x - pad,
      tileY: tile.y - pad,
      frame: job.sample,
      cam: job.cam,
      cam2: job.cam2,
      activeCam: job.activeCam,
      gizmo: null,
      selected: -1,
      outlines: false,
    });

    const write = 1 - job.ping;
    this.accumulatePass(this.exSample.tex, this.exAcc[job.ping].tex, this.exAcc[write].fbo, pw, ph, job.sample);
    job.ping = write;
    job.sample++;
    job.doneSamples++;

    // Stretch the current tile accumulation across the viewport for feedback.
    this.displayPass(null, 0, 0, this.canvas.width, this.canvas.height, this.exAcc[job.ping].tex, null, {
      srcW: pw,
      srcH: ph,
      exposure: state.env.exposure,
      bloomStrength: 0,
      vignette: 0.32,
      denoise: false,
    });

    const now = performance.now();
    if (now - job.lastUi > 90 || job.sample === job.samples) {
      job.lastUi = now;
      const elapsed = now - job.started;
      const per = elapsed / job.doneSamples;
      const remaining = (job.totalSamples - job.doneSamples) * per;
      if (this.onExportProgress) {
        this.onExportProgress({
          tileIndex: job.index,
          tileCount: job.tiles.length,
          sample: job.sample,
          samples: job.samples,
          done: job.doneSamples,
          total: job.totalSamples,
          etaMs: remaining,
        });
      }
    }

    if (job.sample >= job.samples) {
      this._finishTile(job, tile);
      job.index++;
      job.sample = 0;
      if (job.index >= job.tiles.length) {
        this._completeExport(job);
        return;
      }
      // re-upload nothing; object data persists
    }
  }

  _denoisePass(srcTex, dst, w, h, step, sigma, chroma, strength) {
    const gl = this.gl;
    const p = this.progDenoise;
    gl.useProgram(p.prog);
    gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fbo);
    gl.viewport(0, 0, w, h);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, srcTex);
    gl.uniform1i(p.loc('u_tex'), 0);
    gl.uniform2f(p.loc('u_texel'), 1 / w, 1 / h);
    gl.uniform1i(p.loc('u_step'), Math.max(1, step));
    gl.uniform1f(p.loc('u_sigma'), sigma);
    gl.uniform1f(p.loc('u_chroma'), chroma);
    gl.uniform1f(p.loc('u_strength'), strength);
    this._bindQuad(p);
  }

  _finishTile(job, tile) {
    const pad = job.pad;
    const pw = tile.w + pad * 2;
    const ph = tile.h + pad * 2;
    let tex = this.exAcc[job.ping].tex;
    const mode = job.denoise;
    if (mode.passes > 0) {
      for (let i = 0; i < mode.steps.length; i++) {
        const dst = this.exDenoise[i % 2];
        this._denoisePass(tex, dst, pw, ph, mode.steps[i], mode.sigma, mode.chroma, mode.strength);
        tex = dst.tex;
      }
    }
    // The viewport starts at -pad so the filter's padded border is cropped off.
    this.displayPass(this.exDisplay, -pad, -pad, pw, ph, tex, null, {
      srcW: pw,
      srcH: ph,
      exposure: job.exposure ?? 1,
      bloomStrength: 0,
      vignette: 0.35,
      denoise: false,
      tileOrigin: [tile.x, tile.y],
      imageSize: [job.width, job.height],
    });

    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.exDisplay.fbo);
    const px = new Uint8Array(tile.w * tile.h * 4);
    gl.readPixels(0, 0, tile.w, tile.h, gl.RGBA, gl.UNSIGNED_BYTE, px);

    const img = job.ctx.createImageData(tile.w, tile.h);
    const row = tile.w * 4;
    for (let y = 0; y < tile.h; y++) {
      img.data.set(px.subarray((tile.h - 1 - y) * row, (tile.h - y) * row), y * row);
    }
    const tc = job.tileCanvas;
    tc.width = tile.w;
    tc.height = tile.h;
    const tctx = tc.getContext('2d');
    tctx.putImageData(img, 0, 0);
    job.ctx.drawImage(tc, tile.x, job.height - tile.y - tile.h);
  }

  _completeExport(job) {
    job.finishing = true;
    const { canvas, format, width, height } = job;
    const name = `RTD4_render_${width}x${height}_${Date.now()}`;
    const finish = (blob, ext) => {
      const filename = blob ? `${name}.${ext}` : null;
      if (blob) downloadBlob(blob, filename);
      this.exportJob = null;
      this.invalidate();
      if (this.onExportDone) this.onExportDone({ cancelled: false, filename, error: !blob });
    };
    if (format === 'bmp') {
      try {
        const data = job.ctx.getImageData(0, 0, width, height);
        finish(this._encodeBMP(data), 'bmp');
      } catch (err) {
        console.error(err);
        finish(null, 'bmp');
      }
      return;
    }
    const mime = format === 'jpg' ? 'image/jpeg' : format === 'webp' ? 'image/webp' : 'image/png';
    canvas.toBlob((blob) => finish(blob, format), mime, 0.96);
  }

  _encodeBMP(image) {
    const w = image.width;
    const h = image.height;
    const rowSize = (w * 3 + 3) & ~3;
    const pixelSize = rowSize * h;
    const fileSize = 54 + pixelSize;
    const buf = new ArrayBuffer(fileSize);
    const dv = new DataView(buf);
    const bytes = new Uint8Array(buf);
    dv.setUint8(0, 0x42);
    dv.setUint8(1, 0x4d);
    dv.setUint32(2, fileSize, true);
    dv.setUint32(10, 54, true);
    dv.setUint32(14, 40, true);
    dv.setInt32(18, w, true);
    dv.setInt32(22, h, true);
    dv.setUint16(26, 1, true);
    dv.setUint16(28, 24, true);
    dv.setUint32(34, pixelSize, true);
    for (let y = 0; y < h; y++) {
      const src = (h - 1 - y) * w * 4;
      let dst = 54 + y * rowSize;
      for (let x = 0; x < w; x++) {
        const i = src + x * 4;
        bytes[dst++] = image.data[i + 2];
        bytes[dst++] = image.data[i + 1];
        bytes[dst++] = image.data[i];
      }
    }
    return new Blob([buf], { type: 'image/bmp' });
  }

  // Sanity check for the final-render denoiser: run the selected preset over
  // the current accumulation and compare patch variance before and after.
  debugDenoise(mode = 'balanced') {
    if (!this.accTargets || !this.width || !this.height) return null;
    const m = denoiseMode(mode);
    if (m.passes === 0) return null;
    const w = this.width;
    const h = this.height;
    this.ensureExport(w, h);
    const before = this._readVariance(this.accTargets[this.ping].fbo, w, h);
    let tex = this.accTargets[this.ping].tex;
    for (let i = 0; i < m.steps.length; i++) {
      const dst = this.exDenoise[i % 2];
      this._denoisePass(tex, dst, w, h, m.steps[i], m.sigma, m.chroma, m.strength);
      tex = dst.tex;
    }
    const after = this._readVariance(this.exDenoise[(m.steps.length - 1) % 2].fbo, w, h);
    this.invalidate();
    return { mode, before, after };
  }

  _readVariance(fbo, w, h) {
    const gl = this.gl;
    const n = Math.max(4, Math.min(64, w, h));
    const px = new Float32Array(n * n * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.readPixels(Math.floor(w / 2 - n / 2), Math.floor(h / 2 - n / 2), n, n, gl.RGBA, gl.FLOAT, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    let sum = 0;
    let sum2 = 0;
    for (let i = 0; i < n * n; i++) {
      const l = px[i * 4] * 0.2126 + px[i * 4 + 1] * 0.7152 + px[i * 4 + 2] * 0.0722;
      sum += l;
      sum2 += l * l;
    }
    const mean = sum / (n * n);
    return { mean, variance: Math.max(0, sum2 / (n * n) - mean * mean) };
  }

  // Useful for automated sanity checks: sample HDR accumulation in the centre.
  debugLuminance() {
    if (!this.accTargets) return null;
    const gl = this.gl;
    const w = 8;
    const h = 8;
    const px = new Float32Array(w * h * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accTargets[this.ping].fbo);
    gl.readPixels(Math.floor(this.width / 2) - 4, Math.floor(this.height / 2) - 4, w, h, gl.RGBA, gl.FLOAT, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    let sum = 0;
    let bad = false;
    for (let i = 0; i < w * h; i++) {
      const v = px[i * 4] + px[i * 4 + 1] + px[i * 4 + 2];
      if (!isFinite(v)) bad = true;
      sum += Math.max(0, Math.min(v, 10));
    }
    return { mean: sum / (w * h), nan: bad, samples: this.accum, size: [this.width, this.height] };
  }
}
