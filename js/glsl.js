// GLSL ES 3.00 sources for RTD4. All rendering is software ray tracing in a
// WebGL2 fragment shader — no hardware RT, no WASM, no compute shaders.

export const quadVS = `#version 300 es
in vec2 a_position;
out vec2 v_uv;
void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

export const pathTraceFS = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

#define MAX_OBJ 96
#define MAX_LIGHTS 12
#define INF 9999.0
#define PI 3.14159265359

out vec4 fragColor;

uniform sampler2D u_objTex;

uniform vec2 u_resolution;
uniform vec2 u_fullResolution;
uniform vec2 u_tileOffset;
uniform int u_frame;
uniform int u_bounces;
uniform int u_volSteps;

uniform float u_fogDensity;
uniform float u_fogHeight;
uniform float u_fogAniso;
uniform float u_fogGlow;
uniform float u_sunRadius;

uniform int u_objCount;
uniform int u_lightCount;
uniform int u_lights[MAX_LIGHTS];
uniform int u_selected;
uniform int u_showOutlines;

uniform vec3 u_sunDir;
uniform vec3 u_sunColor;
uniform vec3 u_skyColor;
uniform vec3 u_horizonColor;
uniform vec3 u_floorColor;
uniform int u_floorStyle;
uniform float u_floorGloss;
uniform int u_stars;

uniform int u_gizmoMode;
uniform int u_gizmoHot;
uniform float u_gizmoScale;
uniform vec3 u_gizmoPos;

uniform int u_activeCam;
uniform vec3 u_camPos;
uniform vec3 u_camFwd;
uniform vec3 u_camRight;
uniform vec3 u_camUp;
uniform vec3 u_camColor;
uniform float u_camFov;
uniform float u_camAperture;
uniform float u_camFocus;
uniform float u_camVisible;

uniform int u_cam2Active;
uniform vec3 u_cam2Pos;
uniform vec3 u_cam2Fwd;
uniform vec3 u_cam2Right;
uniform vec3 u_cam2Up;
uniform vec3 u_cam2Color;
uniform float u_cam2Visible;

// ---------------- random ----------------
uint lcg(inout uint s) {
  s = s * 1664525u + 1013904223u;
  return s;
}
float rnd(inout uint s) {
  return float(lcg(s) >> 8u) * (1.0 / 16777216.0);
}
vec3 randDir(inout uint s) {
  float z = rnd(s) * 2.0 - 1.0;
  float a = rnd(s) * 6.28318530718;
  float r = sqrt(max(0.0, 1.0 - z * z));
  return vec3(r * cos(a), r * sin(a), z);
}
float hash13(vec3 p) {
  p = fract(p * 0.1031);
  p += dot(p, p.zyx + 31.32);
  return fract((p.x + p.y) * p.z);
}

struct Ray { vec3 ro; vec3 rd; };

struct Hit {
  float t;
  vec3 n;
  vec3 col;
  float prop;
  int mat;
  int id;
  int type;
};

vec4 objT(int i, int k) {
  return texelFetch(u_objTex, ivec2(k, i), 0);
}

// ---------------- intersections ----------------
float sphereHit(vec3 ro, vec3 rd, vec3 c, float r, out vec3 n) {
  vec3 oc = ro - c;
  float b = dot(oc, rd);
  float cc = dot(oc, oc) - r * r;
  float h = b * b - cc;
  if (h <= 0.0) { n = vec3(0.0, 1.0, 0.0); return -1.0; }
  float s = sqrt(h);
  float t = -b - s;
  if (t <= 0.001) t = -b + s;
  if (t <= 0.001) { n = vec3(0.0, 1.0, 0.0); return -1.0; }
  n = normalize(ro + rd * t - c);
  return t;
}

float boxHit(vec3 ro, vec3 rd, vec3 c, vec3 h, out vec3 n) {
  vec3 inv = 1.0 / (rd + vec3(1e-9));
  vec3 t0 = (c - h - ro) * inv;
  vec3 t1 = (c + h - ro) * inv;
  vec3 tmin = min(t0, t1);
  vec3 tmax = max(t0, t1);
  float tn = max(max(tmin.x, tmin.y), tmin.z);
  float tf = min(min(tmax.x, tmax.y), tmax.z);
  if (tn <= 0.001 || tn >= tf) { n = vec3(0.0, 1.0, 0.0); return -1.0; }
  vec3 p = ro + rd * tn - c;
  vec3 a = abs(p) / max(h, vec3(1e-4));
  if (a.x > a.y && a.x > a.z) n = vec3(sign(p.x), 0.0, 0.0);
  else if (a.y > a.z) n = vec3(0.0, sign(p.y), 0.0);
  else n = vec3(0.0, 0.0, sign(p.z));
  return tn;
}

float cylinderHit(vec3 ro, vec3 rd, vec3 base, vec3 axis, float len, float r, out vec3 n) {
  vec3 oc = ro - base;
  float da = dot(rd, axis);
  float oa = dot(oc, axis);
  vec3 rdp = rd - da * axis;
  vec3 ocp = oc - oa * axis;
  float a = dot(rdp, rdp);
  float b = 2.0 * dot(rdp, ocp);
  float c = dot(ocp, ocp) - r * r;
  float disc = b * b - 4.0 * a * c;
  float best = -1.0;
  if (disc > 0.0 && abs(a) > 1e-9) {
    float s = sqrt(disc);
    float t0 = (-b - s) / (2.0 * a);
    float t1 = (-b + s) / (2.0 * a);
    for (int i = 0; i < 2; i++) {
      float t = (i == 0) ? t0 : t1;
      if (t > 0.001) {
        float u = oa + t * da;
        if (u >= 0.0 && u <= len) { best = t; break; }
      }
    }
  }
  if (best <= 0.0) { n = vec3(0.0, 1.0, 0.0); return -1.0; }
  vec3 p = ro + rd * best;
  vec3 rad = (p - base) - axis * dot(p - base, axis);
  n = length(rad) > 1e-5 ? normalize(rad) : axis;
  return best;
}

float coneHit(vec3 ro, vec3 rd, vec3 base, vec3 axis, float len, float r, out vec3 n) {
  vec3 oc = ro - base;
  float da = dot(rd, axis);
  float oa = dot(oc, axis);
  vec3 rdp = rd - da * axis;
  vec3 ocp = oc - oa * axis;
  float A = dot(rdp, rdp);
  float B = 2.0 * dot(rdp, ocp);
  float C = dot(ocp, ocp);
  float k = r / max(len, 1e-4);
  float u0 = 1.0 - oa / max(len, 1e-4);
  float du = -da / max(len, 1e-4);
  float qa = A - k * k * du * du;
  float qb = B - 2.0 * k * k * u0 * du;
  float qc = C - k * k * u0 * u0;
  float disc = qb * qb - 4.0 * qa * qc;
  float best = -1.0;
  if (abs(qa) > 1e-9 && disc > 0.0) {
    float s = sqrt(disc);
    float t0 = (-qb - s) / (2.0 * qa);
    float t1 = (-qb + s) / (2.0 * qa);
    for (int i = 0; i < 2; i++) {
      float t = (i == 0) ? t0 : t1;
      if (t > 0.001) {
        float u = oa + t * da;
        if (u >= 0.0 && u <= len) { best = t; break; }
      }
    }
  }
  if (best <= 0.0) { n = vec3(0.0, 1.0, 0.0); return -1.0; }
  vec3 p = ro + rd * best;
  vec3 rad = (p - base) - axis * dot(p - base, axis);
  vec3 radn = length(rad) > 1e-5 ? normalize(rad) : vec3(1.0, 0.0, 0.0);
  n = normalize(radn + axis * k);
  return best;
}

// ---------------- camera body mesh ----------------
void camBody(Ray r, vec3 pos, vec3 rt, vec3 up, vec3 fw, vec3 col, int camId, int bounce, inout Hit h) {
  if (camId == u_activeCam && bounce == 0) return;
  if (length(pos - r.ro) < 0.5) return;
  vec3 rel = r.ro - pos;
  vec3 lo = vec3(dot(rel, rt), dot(rel, up), dot(rel, fw));
  vec3 ld = vec3(dot(r.rd, rt), dot(r.rd, up), dot(r.rd, fw));
  vec3 n;
  float t = boxHit(lo, ld, vec3(0.0, 0.0, -0.28), vec3(0.23, 0.16, 0.17), n);
  if (t > 0.0 && t < h.t) {
    h.t = t;
    h.n = normalize(n.x * rt + n.y * up + n.z * fw);
    h.col = col;
    h.mat = 4;
    h.prop = 0.35;
    h.id = -99;
    h.type = -1;
  }
  t = cylinderHit(lo, ld, vec3(0.0, 0.0, -0.13), vec3(0.0, 0.0, 1.0), 0.3, 0.085, n);
  if (t > 0.0 && t < h.t) {
    h.t = t;
    h.n = normalize(n.x * rt + n.y * up + n.z * fw);
    h.col = vec3(0.05, 0.06, 0.08);
    h.mat = 1;
    h.prop = 0.15;
    h.id = -99;
    h.type = -1;
  }
}

// ---------------- gizmo ----------------
float gizmoHit(Ray ray, out vec3 gcol) {
  gcol = vec3(0.0);
  if (u_gizmoMode == 0) return -1.0;
  float S = u_gizmoScale;
  float shaftLen = S;
  float shaftR = S * 0.032;
  float tipLen = S * 0.3;
  float tipR = S * 0.092;
  vec3 ax0 = vec3(1.0, 0.0, 0.0);
  vec3 ax1 = vec3(0.0, 1.0, 0.0);
  vec3 ax2 = vec3(0.0, 0.0, 1.0);
  float best = -1.0;
  int bestAxis = 0;
  vec3 bestN = vec3(0.0, 1.0, 0.0);
  vec3 n;

  float t = cylinderHit(ray.ro, ray.rd, u_gizmoPos, ax0, shaftLen, shaftR, n);
  if (t > 0.0) { best = t; bestAxis = 1; bestN = n; }
  t = coneHit(ray.ro, ray.rd, u_gizmoPos + ax0 * shaftLen, ax0, tipLen, tipR, n);
  if (t > 0.0 && (best < 0.0 || t < best)) { best = t; bestAxis = 1; bestN = n; }

  t = cylinderHit(ray.ro, ray.rd, u_gizmoPos, ax1, shaftLen, shaftR, n);
  if (t > 0.0 && (best < 0.0 || t < best)) { best = t; bestAxis = 2; bestN = n; }
  t = coneHit(ray.ro, ray.rd, u_gizmoPos + ax1 * shaftLen, ax1, tipLen, tipR, n);
  if (t > 0.0 && (best < 0.0 || t < best)) { best = t; bestAxis = 2; bestN = n; }

  t = cylinderHit(ray.ro, ray.rd, u_gizmoPos, ax2, shaftLen, shaftR, n);
  if (t > 0.0 && (best < 0.0 || t < best)) { best = t; bestAxis = 3; bestN = n; }
  t = coneHit(ray.ro, ray.rd, u_gizmoPos + ax2 * shaftLen, ax2, tipLen, tipR, n);
  if (t > 0.0 && (best < 0.0 || t < best)) { best = t; bestAxis = 3; bestN = n; }

  t = boxHit(ray.ro, ray.rd, u_gizmoPos, vec3(S * 0.105), n);
  if (t > 0.0 && (best < 0.0 || t < best)) { best = t; bestAxis = 4; bestN = n; }

  if (best <= 0.0) return -1.0;
  vec3 base = bestAxis == 1 ? vec3(1.0, 0.3, 0.32)
            : bestAxis == 2 ? vec3(0.42, 1.0, 0.45)
            : bestAxis == 3 ? vec3(0.38, 0.62, 1.0)
                            : vec3(0.93, 0.95, 1.0);
  float shade = 0.62 + 0.38 * max(dot(bestN, -ray.rd), 0.0);
  if (u_gizmoHot == bestAxis) { base = mix(base, vec3(1.0), 0.45); shade = 1.45; }
  gcol = base * shade;
  return best;
}

// ---------------- scene map ----------------
Hit map(Ray ray, int bounce) {
  Hit h;
  h.t = INF;
  h.n = vec3(0.0, 1.0, 0.0);
  h.col = vec3(0.6);
  h.prop = 0.4;
  h.mat = 0;
  h.id = -2;
  h.type = -1;

  if (abs(ray.rd.y) > 1e-6) {
    float t = -ray.ro.y / ray.rd.y;
    if (t > 0.002 && t < h.t) {
      h.t = t;
      h.n = vec3(0.0, 1.0, 0.0);
      h.col = u_floorColor;
      h.mat = 0;
      h.prop = u_floorGloss;
      h.id = -1;
      h.type = -1;
    }
  }

  for (int i = 0; i < MAX_OBJ; i++) {
    if (i >= u_objCount) break;
    vec4 t0 = objT(i, 0);
    vec4 t1 = objT(i, 1);
    vec4 t2 = objT(i, 2);
    vec3 n;
    float t;
    if (t0.w < 0.5) {
      t = boxHit(ray.ro, ray.rd, t0.xyz, max(t1.xyz * 0.5, vec3(1e-4)), n);
    } else {
      t = sphereHit(ray.ro, ray.rd, t0.xyz, max(t1.x * 0.5, 1e-4), n);
    }
    if (t > 0.0 && t < h.t) {
      h.t = t;
      h.n = n;
      h.col = t2.rgb;
      h.prop = t2.w;
      h.mat = int(t1.w);
      h.id = i;
      h.type = int(t0.w);
    }
  }

  if (u_camVisible > 0.5) {
    camBody(ray, u_camPos, u_camRight, u_camUp, u_camFwd, u_camColor, 0, bounce, h);
  }
  if (u_cam2Active == 1 && u_cam2Visible > 0.5) {
    camBody(ray, u_cam2Pos, u_cam2Right, u_cam2Up, u_cam2Fwd, u_cam2Color, 1, bounce, h);
  }
  return h;
}

// ---------------- lighting helpers ----------------
vec3 floorAlbedo(vec3 p) {
  vec3 base = u_floorColor;
  if (u_floorStyle == 1) {
    float c = mod(floor(p.x) + floor(p.z), 2.0);
    base *= (c > 0.5) ? 1.0 : 0.55;
  } else if (u_floorStyle == 2) {
    vec2 f = abs(fract(p.xz) - 0.5);
    float line = smoothstep(0.452, 0.495, max(f.x, f.y));
    base = mix(base * 0.42, base * 1.2, line);
  }
  return base;
}

vec3 envColor(vec3 rd) {
  float y = clamp(rd.y, -1.0, 1.0);
  vec3 sky = mix(u_horizonColor, u_skyColor, smoothstep(-0.02, 0.72, y));
  if (y < 0.0) sky = mix(sky, u_horizonColor * 0.55, smoothstep(0.0, -0.35, y));
  float sd = dot(rd, u_sunDir);
  sky += u_sunColor * pow(max(sd, 0.0), 24.0) * 0.2;
  float cosO = cos(u_sunRadius * 3.2);
  float cosI = cos(u_sunRadius * 0.55);
  sky += u_sunColor * smoothstep(cosO, cosI, sd) * 2.4;
  float night = smoothstep(0.09, -0.05, u_sunDir.y);
  if (u_stars == 1 && night > 0.01 && y > -0.02) {
    vec3 cell = floor(normalize(rd) * 220.0);
    float h = hash13(cell);
    if (h > 0.9915) {
      float twinkle = 0.65 + 0.35 * hash13(cell + 17.0);
      sky += vec3(0.72, 0.8, 1.0) * (h - 0.9915) * 118.0 * twinkle * night;
    }
  }
  return sky;
}

float hgPhase(float cosT, float g) {
  float g2 = g * g;
  return (1.0 - g2) / (12.5663706 * pow(max(1e-4, 1.0 + g2 - 2.0 * g * cosT), 1.5));
}

vec3 sampleLights(vec3 p, vec3 n, vec3 albedo, inout uint seed) {
  if (u_lightCount <= 0) return vec3(0.0);
  int li = int(rnd(seed) * float(u_lightCount));
  li = min(li, u_lightCount - 1);
  int idx = u_lights[li];
  vec4 a = objT(idx, 0);
  vec4 b = objT(idx, 1);
  vec4 c = objT(idx, 2);
  float rad = max(0.5 * (a.w < 0.5 ? max(b.x, max(b.y, b.z)) : b.x), 0.04);
  vec3 lp = a.xyz + randDir(seed) * rad;
  vec3 L = lp - p;
  float d2 = dot(L, L);
  if (d2 < 1e-4) return vec3(0.0);
  float d = sqrt(d2);
  vec3 l = L / d;
  float cs = dot(n, l);
  if (cs <= 0.0) return vec3(0.0);
  vec3 ln = normalize(lp - a.xyz);
  float cl = max(dot(ln, -l), 0.0);
  if (cl <= 0.0) return vec3(0.0);
  Hit sh = map(Ray(p + n * 0.003, l), 99);
  if (sh.t < d - rad * 0.6 && sh.mat != 2) return vec3(0.0);
  vec3 emit = c.rgb * max(c.w, 0.0);
  return albedo * emit * cs * cl * (4.0 * rad * rad) / d2;
}

vec3 fogSample(Ray ray, float maxDist, inout uint seed, out float trans) {
  trans = 1.0;
  if (u_fogDensity <= 0.0001 || u_volSteps <= 0) return vec3(0.0);
  float dist = min(maxDist, 60.0);
  if (dist <= 0.001) return vec3(0.0);
  int steps = u_volSteps;
  float stepSize = dist / float(steps);
  float t = rnd(seed) * stepSize;
  vec3 acc = vec3(0.0);
  float phase = hgPhase(dot(ray.rd, u_sunDir), clamp(u_fogAniso, -0.95, 0.95));

  int lightIdx = -1;
  vec3 lightPos = vec3(0.0);
  vec3 lightEmit = vec3(0.0);
  float lightRad = 0.0;
  if (u_lightCount > 0 && u_fogGlow > 0.001) {
    int li = int(rnd(seed) * float(u_lightCount));
    li = min(li, u_lightCount - 1);
    lightIdx = u_lights[li];
    vec4 a = objT(lightIdx, 0);
    vec4 b = objT(lightIdx, 1);
    vec4 c = objT(lightIdx, 2);
    lightPos = a.xyz;
    lightRad = max(0.5 * (a.w < 0.5 ? max(b.x, max(b.y, b.z)) : b.x), 0.04);
    lightEmit = c.rgb * max(c.w, 0.0);
  }

  for (int i = 0; i < 32; i++) {
    if (i >= steps || t > dist) break;
    vec3 p = ray.ro + ray.rd * t;
    float dens = u_fogDensity * exp(-max(p.y, 0.0) * u_fogHeight);
    if (dens > 1e-5) {
      float ds = dens * stepSize;
      if (dot(u_sunColor, u_sunColor) > 1e-4) {
        vec3 sd = normalize(u_sunDir + randDir(seed) * u_sunRadius * 2.4);
        Hit sh = map(Ray(p + sd * 0.01, sd), 99);
        if (sh.t >= INF || sh.mat == 2) acc += trans * u_sunColor * ds * phase * 3.0;
      }
      acc += trans * u_skyColor * ds * 0.32;
      if (lightIdx >= 0) {
        vec3 L = lightPos - p;
        float d2 = dot(L, L);
        float d = sqrt(d2);
        vec3 ld = L / max(d, 1e-4);
        Hit sh = map(Ray(p + ld * 0.01, ld), 99);
        if (sh.t >= d - lightRad) {
          acc += trans * lightEmit * (ds * 8.0 * u_fogGlow) / (d2 + 0.35);
        }
      }
      trans *= exp(-dens * stepSize);
      if (trans < 0.012) break;
    }
    t += stepSize;
  }
  return acc;
}

// ---------------- path tracer ----------------
vec3 trace(Ray ray, inout uint seed) {
  vec3 col = vec3(0.0);
  vec3 thr = vec3(1.0);

  for (int bounce = 0; bounce < 8; bounce++) {
    if (bounce >= u_bounces) break;
    Hit h = map(ray, bounce);

    if (bounce == 0 && u_gizmoMode > 0) {
      vec3 gc;
      float gt = gizmoHit(ray, gc);
      if (gt > 0.0) return col + gc;
    }

    float fogT = 1.0;
    if (bounce == 0 && u_fogDensity > 0.0001) {
      float fd = min(h.t, 60.0);
      vec3 fs = fogSample(ray, fd, seed, fogT);
      col += thr * fs;
      thr *= fogT;
    }

    if (h.t >= INF) {
      col += thr * envColor(ray.rd);
      break;
    }

    vec3 p = ray.ro + ray.rd * h.t;
    vec3 n = h.n;
    vec3 albedo = (h.id == -1) ? floorAlbedo(p) : h.col;
    float outlineMask = 0.0;

    if (bounce == 0 && u_showOutlines == 1 && h.id == u_selected && h.id >= 0) {
      if (h.type == 0) {
        vec4 t0 = objT(h.id, 0);
        vec4 t1 = objT(h.id, 1);
        vec3 hs = max(t1.xyz * 0.5, vec3(1e-3));
        vec3 lp = (p - t0.xyz) / hs;
        vec3 an = abs(n);
        float du, dv, hu, hv;
        if (an.y > 0.5) { du = 1.0 - abs(lp.x); dv = 1.0 - abs(lp.z); hu = hs.x; hv = hs.z; }
        else if (an.x > 0.5) { du = 1.0 - abs(lp.y); dv = 1.0 - abs(lp.z); hu = hs.y; hv = hs.z; }
        else { du = 1.0 - abs(lp.x); dv = 1.0 - abs(lp.y); hu = hs.x; hv = hs.y; }
        float edge = min(du * hu, dv * hv);
        outlineMask = 1.0 - smoothstep(0.0, 0.03 + 0.02 * max(hu, hv), edge);
      } else {
        vec4 t1 = objT(h.id, 1);
        float rim = 1.0 - max(dot(-ray.rd, n), 0.0);
        outlineMask = smoothstep(0.68, 0.9, rim) * (1.0 - smoothstep(0.9, 1.6, h.t / max(t1.x, 0.2)));
      }
      albedo = mix(albedo, vec3(1.0, 0.78, 0.2), outlineMask * 0.92);
    }

    if (h.mat == 3) {
      vec3 em = h.col * max(h.prop, 0.0);
      em = mix(em, vec3(1.0, 0.82, 0.3) * 2.0, outlineMask);
      col += thr * em;
      break;
    }

    if (dot(u_sunColor, u_sunColor) > 1e-5 && (h.mat == 0 || h.mat == 4 || h.id == -1)) {
      vec3 sd = normalize(u_sunDir + randDir(seed) * u_sunRadius * 1.5);
      float ndl = dot(n, sd);
      if (ndl > 0.0) {
        Hit sh = map(Ray(p + n * 0.003, sd), 99);
        if (sh.t >= INF || sh.mat == 2) {
          col += thr * albedo * u_sunColor * ndl;
          vec3 hv = normalize(sd - ray.rd);
          float shin = (h.mat == 4) ? mix(18.0, 240.0, 1.0 - clamp(h.prop, 0.0, 1.0)) : mix(6.0, 48.0, 1.0 - clamp(h.prop, 0.0, 1.0));
          float spec = pow(max(dot(n, hv), 0.0), shin) * (h.mat == 4 ? 0.55 : 0.16);
          col += thr * u_sunColor * spec * ndl * (0.35 + 0.65 * albedo);
        }
      }
    }

    if (u_lightCount > 0 && (h.mat == 0 || h.mat == 4 || h.id == -1)) {
      col += thr * sampleLights(p, n, albedo, seed);
    }

    if (h.mat == 1) {
      vec3 r = reflect(ray.rd, n);
      r = normalize(r + randDir(seed) * (h.prop * h.prop) * 0.85);
      if (dot(r, n) <= 0.0) r = n;
      ray = Ray(p + n * 0.003, r);
      thr *= mix(albedo, vec3(1.0), 0.08);
    } else if (h.mat == 2) {
      float ior = clamp(h.prop, 1.01, 3.0);
      vec3 outN = n;
      float eta = 1.0 / ior;
      if (dot(ray.rd, n) > 0.0) { outN = -n; eta = ior; }
      float cosi = max(0.0, dot(-ray.rd, outN));
      float r0 = (1.0 - ior) / (1.0 + ior);
      r0 *= r0;
      float f = r0 + (1.0 - r0) * pow(1.0 - cosi, 5.0);
      if (rnd(seed) < f) {
        ray = Ray(p + outN * 0.003, reflect(ray.rd, outN));
      } else {
        vec3 rr = refract(ray.rd, outN, eta);
        if (dot(rr, rr) < 1e-6) ray = Ray(p + outN * 0.003, reflect(ray.rd, outN));
        else ray = Ray(p - outN * 0.003, normalize(rr));
      }
      thr *= mix(vec3(1.0), albedo, 0.3);
    } else if (h.mat == 4) {
      if (rnd(seed) < 0.34) {
        vec3 r = normalize(reflect(ray.rd, n) + randDir(seed) * (0.04 + h.prop * h.prop * 0.55));
        if (dot(r, n) <= 0.0) r = n;
        ray = Ray(p + n * 0.003, r);
        thr *= mix(vec3(1.0), albedo, 0.45) * 0.72;
      } else {
        ray = Ray(p + n * 0.003, normalize(n + randDir(seed)));
        thr *= albedo * 0.92;
      }
    } else {
      if (h.id == -1 && u_floorGloss > 0.001 && rnd(seed) < u_floorGloss) {
        vec3 r = normalize(reflect(ray.rd, n) + randDir(seed) * 0.035);
        if (dot(r, n) <= 0.0) r = n;
        ray = Ray(p + n * 0.003, r);
        thr *= mix(albedo, vec3(1.0), 0.8) * 0.55;
      } else {
        ray = Ray(p + n * 0.003, normalize(n + randDir(seed)));
        thr *= albedo * 0.95;
      }
    }

    thr = clamp(thr, 0.0, 4.0);
    if (bounce > 2) {
      float q = max(thr.r, max(thr.g, thr.b));
      if (rnd(seed) > q) break;
      thr /= max(q, 0.001);
    }
  }
  return col;
}

void main() {
  vec2 pix = gl_FragCoord.xy + u_tileOffset;
  uint seed = uint(pix.x) * 1973u + uint(pix.y) * 9277u + uint(u_frame) * 26699u + 197u;
  seed ^= seed << 13u;
  seed *= 2654435761u;
  seed += 1u;

  float jx = (rnd(seed) - 0.5) * 2.0 / u_fullResolution.x;
  float jy = (rnd(seed) - 0.5) * 2.0 / u_fullResolution.y;
  float aspect = u_fullResolution.x / max(u_fullResolution.y, 1.0);
  vec2 uv = (pix / u_fullResolution) * 2.0 - 1.0;
  uv.x *= aspect;
  uv += vec2(jx * aspect, jy);

  vec3 dir = normalize(u_camFwd + u_camRight * (uv.x * u_camFov) + u_camUp * (uv.y * u_camFov));
  vec3 origin = u_camPos;

  if (u_camAperture > 0.0004) {
    vec3 focusPoint = u_camPos + dir * max(u_camFocus, 0.25);
    vec3 off = (u_camRight * (rnd(seed) * 2.0 - 1.0) + u_camUp * (rnd(seed) * 2.0 - 1.0)) * u_camAperture;
    origin += off;
    dir = normalize(focusPoint - origin);
  }

  fragColor = vec4(trace(Ray(origin, dir), seed), 1.0);
}
`;

export const accumulateFS = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_current;
uniform sampler2D u_history;
uniform float u_weight;
void main() {
  vec3 cur = texture(u_current, v_uv).rgb;
  vec3 hist = texture(u_history, v_uv).rgb;
  outColor = vec4(mix(hist, cur, clamp(u_weight, 0.0, 1.0)), 1.0);
}
`;

export const prefilterFS = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_tex;
uniform float u_threshold;
void main() {
  vec3 c = texture(u_tex, v_uv).rgb;
  float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
  float k = max(l - u_threshold, 0.0) / max(l, 1e-4);
  outColor = vec4(c * k, 1.0);
}
`;

export const blurFS = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_tex;
uniform vec2 u_dir;
void main() {
  vec3 sum = vec3(0.0);
  float wsum = 0.0;
  for (int i = -4; i <= 4; i++) {
    float fi = float(i);
    float w = exp(-(fi * fi) / 10.0);
    sum += texture(u_tex, v_uv + u_dir * fi).rgb * w;
    wsum += w;
  }
  outColor = vec4(sum / wsum, 1.0);
}
`;

export const displayFS = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_tex;
uniform sampler2D u_bloom;
uniform vec2 u_texel;
uniform float u_exposure;
uniform float u_bloomStrength;
uniform float u_vignette;
uniform float u_denoise;
uniform vec2 u_tileOrigin;
uniform vec2 u_imageSize;

float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

vec3 aces(vec3 x) {
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
  vec3 c = texture(u_tex, v_uv).rgb;

  if (u_denoise > 0.5) {
    float lc = luma(c);
    vec3 acc = c * 2.2;
    float wsum = 2.2;
    for (int y = -1; y <= 1; y++) {
      for (int x = -1; x <= 1; x++) {
        if (x == 0 && y == 0) continue;
        vec3 s = texture(u_tex, v_uv + vec2(float(x), float(y)) * u_texel).rgb;
        float w = exp(-abs(luma(s) - lc) * 12.0) * exp(-float(x * x + y * y) * 0.85);
        acc += s * w;
        wsum += w;
      }
    }
    c = acc / wsum;
  }

  if (u_bloomStrength > 0.0) c += texture(u_bloom, v_uv).rgb * u_bloomStrength;
  c *= u_exposure;
  c = aces(c);

  // Tiled exports pass the tile's origin and the full image size so the
  // vignette is continuous across tile borders instead of restarting per tile.
  vec2 vuv = (u_imageSize.x > 0.5) ? (gl_FragCoord.xy + u_tileOrigin) / u_imageSize : v_uv;
  vec2 q = vuv - 0.5;
  float vig = smoothstep(1.02, 0.28, length(q * vec2(1.62, 1.0)));
  c *= mix(1.0, vig, u_vignette);

  outColor = vec4(pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}
`;

// Edge-aware a-trous wavelet filter for the final render. It runs on the HDR
// accumulation buffer before tone mapping, so it removes Monte Carlo variance
// without crushing highlight detail or bleeding across object silhouettes.
export const denoiseFS = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_tex;
uniform vec2 u_texel;
uniform int u_step;
uniform float u_sigma;
uniform float u_chroma;
uniform float u_strength;

float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

void main() {
  vec3 c = texture(u_tex, v_uv).rgb;
  float lc = luma(c);
  vec3 acc = c * 0.375;
  float wsum = 0.375;

  for (int y = -2; y <= 2; y++) {
    for (int x = -2; x <= 2; x++) {
      if (x == 0 && y == 0) continue;
      float kx = (x == 0) ? 0.375 : (abs(x) == 1 ? 0.25 : 0.0625);
      float ky = (y == 0) ? 0.375 : (abs(y) == 1 ? 0.25 : 0.0625);
      vec3 s = texture(u_tex, v_uv + vec2(float(x), float(y)) * u_texel * float(u_step)).rgb;
      float ls = luma(s);
      float dl = abs(ls - lc) / (u_sigma + 0.06 * max(lc, ls));
      float dc = length(s - c) / (u_chroma + 0.08 * max(lc, 0.08));
      float w = kx * ky * exp(-dl * 1.6) * exp(-dc * 0.9);
      acc += s * w;
      wsum += w;
    }
  }

  vec3 sm = acc / max(wsum, 1e-4);
  outColor = vec4(mix(c, sm, clamp(u_strength, 0.0, 1.0)), 1.0);
}
`;
