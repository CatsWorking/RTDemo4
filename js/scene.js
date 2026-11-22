import { mulberry32, rgbToHex } from './utils.js?v=6';

export const MAX_OBJECTS = 96;
export const MAX_LIGHTS = 12;

export const MATERIALS = [
  { name: 'Matte', prop: 'Sheen', min: 0, max: 1, step: 0.01, def: 0.45 },
  { name: 'Metal', prop: 'Roughness', min: 0, max: 1, step: 0.01, def: 0.04 },
  { name: 'Glass', prop: 'IOR', min: 1.01, max: 2.8, step: 0.01, def: 1.5 },
  { name: 'Neon', prop: 'Intensity', min: 0.2, max: 10, step: 0.1, def: 3.2 },
  { name: 'Glossy', prop: 'Roughness', min: 0, max: 1, step: 0.01, def: 0.22 },
];

export const DEFAULT_ENV = {
  time: 14.2,
  animate: false,
  sunColor: '#fff2dc',
  fog: 0.012,
  aniso: 0.4,
  fogHeight: 0.12,
  fogGlow: 1,
  volSteps: 10,
  floorColor: '#8a919c',
  floorStyle: 1,
  floorGloss: 0.3,
  exposure: 1,
  bloom: 0.35,
  aperture: 0,
  autoFocus: true,
  focus: 6,
};

let nextName = 1;

export function makeObject(type, pos, size, mat = 0, color = '#ffffff', prop) {
  const s = Array.isArray(size) ? size.map(Number) : [size, size, size];
  const m = Number(mat);
  return {
    type: type === 1 ? 1 : 0,
    pos: pos.map(Number),
    scale: s,
    mat: m,
    color,
    prop: prop !== undefined ? Number(prop) : MATERIALS[m].def,
  };
}

export function objectLabel(obj, index) {
  return `${obj.type === 1 ? 'Sphere' : 'Cube'} ${index + 1}`;
}

export function clampObject(obj) {
  const type = obj.type === 1 ? 1 : 0;
  const pos = Array.isArray(obj.pos)
    ? [0, 1, 2].map((i) => (Number.isFinite(Number(obj.pos[i])) ? Number(obj.pos[i]) : [0, 1, 0][i]))
    : [0, 1, 0];
  let scale = Array.isArray(obj.scale)
    ? [0, 1, 2].map((i) => {
        const v = Number(obj.scale[i]);
        return Number.isFinite(v) ? Math.max(0.1, Math.min(8, v)) : 1;
      })
    : [1, 1, 1];
  if (type === 1) scale = [scale[0], scale[0], scale[0]];
  const mat = Math.max(0, Math.min(4, Number.isFinite(Number(obj.mat)) ? Number(obj.mat) : 0));
  const meta = MATERIALS[mat];
  const prop = Math.max(meta.min, Math.min(meta.max, Number.isFinite(Number(obj.prop)) ? Number(obj.prop) : meta.def));
  const color = typeof obj.color === 'string' && /^#[0-9a-f]{6}$/i.test(obj.color) ? obj.color : '#ffffff';
  return { type, pos, scale, mat, color, prop };
}

const box = (pos, scale, mat, color, prop) => makeObject(0, pos, scale, mat, color, prop);
const sphere = (pos, size, mat, color, prop) => makeObject(1, pos, size, mat, color, prop);

export const PRESETS = {
  classic: {
    label: 'Classic',
    build() {
      return {
        objects: [
          sphere([0, 1.25, 0], 2.5, 1, '#ffffff', 0.035),
          sphere([2.65, 1.05, 1.05], 2.1, 2, '#e6f6ff', 1.46),
          box([-2.65, 1.05, 1.05], 2.1, 3, '#3ed8ff', 2.6),
          box([-0.15, 0.45, -2.4], 0.9, 4, '#ff7ab8', 0.22),
        ],
        env: { time: 14.4, fog: 0.01, aniso: 0.42, volSteps: 10, floorColor: '#8b929d', floorStyle: 1, floorGloss: 0.34, exposure: 1, bloom: 0.35, aperture: 0, focus: 6 },
        camera: { pos: [0.9, 1.85, 7.2], target: [0, 1.15, 0], fov: 78 },
      };
    },
  },

  cornell: {
    label: 'Cornell Box',
    build() {
      return {
        objects: [
          box([-3.5, 2.6, -2.5], [0.24, 5.2, 8], 0, '#c23a2d', 0.4),
          box([3.5, 2.6, -2.5], [0.24, 5.2, 8], 0, '#2f9e58', 0.4),
          box([0, 2.6, -6.45], [7.1, 5.2, 0.24], 0, '#e2ddd3', 0.4),
          box([0, 5.25, -2.5], [7.1, 0.24, 8], 0, '#e2ddd3', 0.4),
          box([0, 5.0, -2.5], [2.4, 0.16, 2.4], 3, '#fff0d2', 7),
          box([-1.35, 1.0, -3.4], 2, 4, '#f4efe6', 0.15),
          sphere([1.45, 0.85, -2.2], 1.7, 1, '#f2f2f2', 0.02),
          sphere([1.15, 0.4, -4.3], 0.8, 2, '#cfe9ff', 1.5),
        ],
        env: { time: 12, fog: 0.0015, aniso: 0.2, volSteps: 8, floorColor: '#b7b0a4', floorStyle: 0, floorGloss: 0.06, exposure: 1.5, bloom: 0.5, aperture: 0, focus: 4 },
        camera: { pos: [0, 1.75, 0.35], target: [0, 2.2, -3.6], fov: 72 },
      };
    },
  },

  neon: {
    label: 'Neon Alley',
    build() {
      const objects = [];
      const colors = ['#ff3fa4', '#37e8ff', '#ffd23f', '#b06bff'];
      for (let i = 0; i < 6; i++) {
        const z = 2.4 - i * 2.2;
        objects.push(box([-2.55, 1.85, z], [0.34, 3.7, 0.34], 3, colors[i % 4], 2.4));
        objects.push(box([2.55, 1.85, z], [0.34, 3.7, 0.34], 3, colors[(i + 2) % 4], 2.4));
      }
      objects.push(sphere([0, 0.8, -1], 1.6, 1, '#ffffff', 0.05));
      objects.push(box([0, 0.45, -5], 0.9, 4, '#ff9fd0', 0.1));
      return {
        objects,
        env: { time: 21.4, fog: 0.045, aniso: 0.72, fogHeight: 0.05, fogGlow: 1.7, volSteps: 14, floorColor: '#3c4350', floorStyle: 2, floorGloss: 0.55, exposure: 1.25, bloom: 0.8, aperture: 0.035, focus: 4 },
        camera: { pos: [0, 1.45, 8.4], target: [0, 1.7, -2], fov: 82 },
      };
    },
  },

  mirror: {
    label: 'Mirror Hall',
    build() {
      const objects = [];
      for (let i = 0; i < 4; i++) {
        const x = -4.2 + i * 2.8;
        objects.push(box([x, 2.3, -6.2], [1.8, 4.6, 0.35], 1, '#f4f7ff', 0.02));
      }
      objects.push(sphere([-2.2, 1.15, -1.5], 2.3, 1, '#ffffff', 0.01));
      objects.push(sphere([2.2, 1.15, -1.5], 2.3, 1, '#ffe2c4', 0.08));
      objects.push(box([0, 0.55, 0.4], 1.1, 4, '#2fe6c8', 0.05));
      objects.push(box([-4.6, 0.4, 2.4], 0.8, 3, '#7fe9ff', 3));
      objects.push(box([4.6, 0.4, 2.4], 0.8, 3, '#ff7fc4', 3));
      return {
        objects,
        env: { time: 10.2, fog: 0.014, aniso: 0.5, volSteps: 12, floorColor: '#a8b4c4', floorStyle: 2, floorGloss: 0.72, exposure: 1.05, bloom: 0.5, aperture: 0, focus: 5 },
        camera: { pos: [0, 1.75, 8.6], target: [0, 1.5, -3], fov: 74 },
      };
    },
  },

  orb: {
    label: 'Orb Field',
    build() {
      const rnd = mulberry32(4411);
      const objects = [];
      const mats = [0, 1, 2, 3, 4];
      const palette = ['#ffd479', '#ff6b9d', '#67e8f9', '#a78bfa', '#86efac'];
      for (let i = 0; i < 14; i++) {
        const a = (i / 14) * Math.PI * 2;
        const r = 1.6 + rnd() * 4.4;
        const size = 0.5 + rnd() * 1.4;
        const mat = mats[Math.floor(rnd() * mats.length)];
        objects.push(sphere([Math.cos(a) * r, size * 0.5 + 0.02, Math.sin(a) * r - 2], size, mat, palette[i % palette.length], mat === 2 ? 1.5 : 0.08 + rnd() * 0.4));
      }
      objects.push(box([0, 0.5, -2], 1, 4, '#e8eef8', 0.06));
      return {
        objects,
        env: { time: 17.15, fog: 0.026, aniso: 0.62, fogHeight: 0.08, fogGlow: 1.3, volSteps: 14, floorColor: '#6c7484', floorStyle: 1, floorGloss: 0.4, exposure: 1.2, bloom: 0.55, aperture: 0.02, focus: 6 },
        camera: { pos: [0, 2.4, 9.6], target: [0, 0.9, -2], fov: 76 },
      };
    },
  },

  studio: {
    label: 'Studio',
    build() {
      return {
        objects: [
          sphere([-1.5, 1.0, 0], 2, 1, '#ffffff', 0.01),
          box([1.35, 0.8, 0.5], [1.6, 1.6, 1.6], 4, '#f4f6fa', 0.12),
          sphere([1.15, 0.65, -1.75], 1.3, 2, '#dff4ff', 1.52),
          box([-0.15, 0.35, -2.8], 0.6, 3, '#ffd7a3', 2.2),
        ],
        env: { time: 13.2, fog: 0.004, aniso: 0.3, volSteps: 8, floorColor: '#2b303a', floorStyle: 0, floorGloss: 0.6, exposure: 1.05, bloom: 0.5, aperture: 0.06, focus: 5.2 },
        camera: { pos: [0, 1.55, 6.1], target: [0, 1.0, 0], fov: 55 },
      };
    },
  },

  dunes: {
    label: 'Dunes',
    build() {
      const objects = [];
      const rnd = mulberry32(90210);
      const palette = ['#c28b53', '#a9713f', '#d9a877', '#b57c48', '#e2b98c'];
      for (let i = 0; i < 12; i++) {
        const x = -9 + rnd() * 18;
        const z = -14 + rnd() * 12;
        const w = 2 + rnd() * 5;
        const h = 0.5 + rnd() * 2.6;
        const d = 2 + rnd() * 5;
        objects.push(box([x, h / 2, z], [w, h, d], rnd() > 0.7 ? 4 : 0, palette[i % palette.length], rnd() * 0.6));
      }
      objects.push(sphere([0, 0.8, 0.5], 1.6, 1, '#fdf3e3', 0.06));
      return {
        objects,
        env: { time: 17.05, fog: 0.038, aniso: 0.7, fogHeight: 0.06, fogGlow: 0.8, volSteps: 16, floorColor: '#b8875a', floorStyle: 0, floorGloss: 0.14, exposure: 1.25, bloom: 0.6, aperture: 0, focus: 8 },
        camera: { pos: [5.5, 2.2, 13], target: [0, 1.3, -2], fov: 62 },
      };
    },
  },

  void: {
    label: 'Empty Void',
    build() {
      return {
        objects: [],
        env: { time: 14, fog: 0, volSteps: 0, floorColor: '#4b5563', floorStyle: 2, floorGloss: 0.35, exposure: 1, bloom: 0.35, aperture: 0, focus: 6 },
        camera: { pos: [0, 2.0, 7], target: [0, 1, 0], fov: 80 },
      };
    },
  },
};

export function remixObjects(objects, seed = Date.now()) {
  const rnd = mulberry32(seed);
  const palette = ['#ff6b9d', '#ffd479', '#67e8f9', '#a78bfa', '#86efac', '#f97362', '#f8fafc'];
  for (const o of objects) {
    const mat = Math.floor(rnd() * 5);
    o.mat = mat;
    o.color = mat === 1 ? (rnd() > 0.5 ? '#ffffff' : palette[Math.floor(rnd() * palette.length)]) : palette[Math.floor(rnd() * palette.length)];
    const meta = MATERIALS[mat];
    o.prop = meta.min + rnd() * (meta.max - meta.min) * (mat === 2 ? 0.35 : 0.8);
    if (mat === 2) o.prop = 1.15 + rnd() * 0.6;
    if (mat === 3) o.prop = 1.5 + rnd() * 4;
    if (mat === 1) o.prop = rnd() * 0.25;
  }
  return objects;
}

export function makeScenePayload(objects, env, camera, renderCamera) {
  return {
    format: 'rtd4-scene',
    version: 4,
    savedAt: new Date().toISOString(),
    objects: objects.map((o) => ({ type: o.type, pos: o.pos, scale: o.scale, mat: o.mat, color: o.color, prop: o.prop })),
    env: { ...env },
    camera: camera.serialize(),
    renderCamera: renderCamera ? renderCamera.serialize() : null,
  };
}

export function parseScenePayload(data) {
  if (!data || typeof data !== 'object' || !Array.isArray(data.objects)) {
    throw new Error('Not an RTD4 scene file');
  }
  const objects = data.objects.slice(0, MAX_OBJECTS).map(clampObject);
  const env = { ...DEFAULT_ENV };
  const num = (v, def) => (Number.isFinite(Number(v)) ? Number(v) : def);
  const hex = (v, def) => (typeof v === 'string' && /^#[0-9a-f]{6}$/i.test(v) ? v : def);
  if (data.env && typeof data.env === 'object') {
    for (const key of Object.keys(DEFAULT_ENV)) {
      if (!(key in data.env)) continue;
      const v = data.env[key];
      const def = DEFAULT_ENV[key];
      if (typeof def === 'boolean') env[key] = !!v;
      else if (typeof def === 'string') env[key] = key.toLowerCase().includes('color') ? hex(v, def) : String(v);
      else env[key] = num(v, def);
    }
  }
  env.time = Math.max(0, Math.min(24, env.time));
  env.fog = Math.max(0, Math.min(0.08, env.fog));
  env.aniso = Math.max(-0.3, Math.min(0.92, env.aniso));
  env.fogHeight = Math.max(0, Math.min(0.6, env.fogHeight));
  env.fogGlow = Math.max(0, Math.min(3, env.fogGlow));
  env.exposure = Math.max(0.2, Math.min(3, env.exposure));
  env.bloom = Math.max(0, Math.min(1.5, env.bloom));
  env.aperture = Math.max(0, Math.min(0.25, env.aperture));
  env.focus = Math.max(0.5, Math.min(40, env.focus));
  env.floorGloss = Math.max(0, Math.min(0.9, env.floorGloss));
  env.volSteps = [0, 6, 8, 10, 12, 14, 16, 20, 24].includes(env.volSteps) ? env.volSteps : 10;
  env.floorStyle = [0, 1, 2].includes(env.floorStyle) ? env.floorStyle : 1;
  return { objects, env, camera: data.camera, renderCamera: data.renderCamera || null };
}

export { rgbToHex };
