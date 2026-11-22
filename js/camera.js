import { clamp, dot3, normalize3, sub3, cross3 } from './utils.js';

export class Camera {
  constructor(opts = {}) {
    this.pos = [...(opts.pos || [0, 2, 6])];
    this.yaw = opts.yaw ?? -Math.PI / 2;
    this.pitch = opts.pitch ?? 0;
    this.fov = opts.fov ?? 90;
    this.visible = opts.visible ?? true;
    this.color = opts.color ?? '#232833';
    this.aperture = 0;
    this.forward = [0, 0, -1];
    this.right = [1, 0, 0];
    this.up = [0, 1, 0];
    this.fovScale = 1;
    this.updateVectors();
  }

  updateVectors() {
    const cp = Math.cos(this.pitch);
    this.forward = normalize3([Math.cos(this.yaw) * cp, Math.sin(this.pitch), Math.sin(this.yaw) * cp]);
    this.right = normalize3(cross3(this.forward, [0, 1, 0]));
    if (!isFinite(this.right[0]) || Math.hypot(...this.right) < 1e-4) this.right = [1, 0, 0];
    this.up = normalize3(cross3(this.right, this.forward));
    this.fovScale = Math.tan((this.fov * Math.PI) / 360);
  }

  lookAt(target) {
    const d = normalize3(sub3(target, this.pos));
    this.pitch = Math.asin(clamp(d[1], -1, 1));
    this.yaw = Math.atan2(d[2], d[0]);
    this.updateVectors();
  }

  setFov(fov) {
    this.fov = fov;
    this.fovScale = Math.tan((fov * Math.PI) / 360);
  }

  move(forwardAxis, strafeAxis, liftAxis, speed, dt) {
    const step = speed * dt;
    this.pos = [
      this.pos[0] + (this.forward[0] * forwardAxis + this.right[0] * strafeAxis) * step,
      this.pos[1] + (this.forward[1] * forwardAxis + this.right[1] * strafeAxis) * step + liftAxis * step,
      this.pos[2] + (this.forward[2] * forwardAxis + this.right[2] * strafeAxis) * step,
    ];
  }

  ray(ndcX, ndcY, aspect) {
    const dir = normalize3([
      this.forward[0] + this.right[0] * ndcX * aspect * this.fovScale + this.up[0] * ndcY * this.fovScale,
      this.forward[1] + this.right[1] * ndcX * aspect * this.fovScale + this.up[1] * ndcY * this.fovScale,
      this.forward[2] + this.right[2] * ndcX * aspect * this.fovScale + this.up[2] * ndcY * this.fovScale,
    ]);
    return { origin: [...this.pos], dir };
  }

  serialize() {
    return {
      pos: [...this.pos],
      yaw: this.yaw,
      pitch: this.pitch,
      fov: this.fov,
      visible: this.visible,
      color: this.color,
    };
  }

  static deserialize(data = {}) {
    const raw = Array.isArray(data.pos) ? data.pos.map(Number) : [];
    const pos = [0, 1, 2].map((i) => (Number.isFinite(raw[i]) ? raw[i] : [0, 2, 6][i]));
    const cam = new Camera({
      pos,
      yaw: Number.isFinite(Number(data.yaw)) ? Number(data.yaw) : 0,
      pitch: clamp(Number.isFinite(Number(data.pitch)) ? Number(data.pitch) : 0, -1.5, 1.5),
      fov: clamp(Number.isFinite(Number(data.fov)) ? Number(data.fov) : 90, 10, 170),
      visible: data.visible !== false,
      color: typeof data.color === 'string' && /^#[0-9a-f]{6}$/i.test(data.color) ? data.color : '#232833',
    });
    return cam;
  }
}

export function cameraDistance(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

export function dot(a, b) {
  return dot3(a, b);
}
