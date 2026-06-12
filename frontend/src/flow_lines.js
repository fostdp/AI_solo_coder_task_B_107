import * as THREE from 'three';

const MAX_STREAMLINES = 80;
const TUBE_SEGMENTS = 20;
const TUBE_RADIUS = 0.004;
const TUBE_RADIAL_SEGMENTS = 4;
const PARTICLES_PER_LINE = 3;

export class FlowLineManager {
    constructor(scene) {
        this.scene = scene;
        this.streamlineGroup = new THREE.Group();
        this.streamlineGroup.userData.type = 'streamlineGroup';
        this.scene.add(this.streamlineGroup);

        this.tubes = [];
        this.particleMesh = null;
        this.particleData = [];
        this.curves = [];
        this.visible = true;

        this._initParticlePool();
    }

    _initParticlePool() {
        const maxParticles = MAX_STREAMLINES * PARTICLES_PER_LINE;
        const positions = new Float32Array(maxParticles * 3);
        const scales = new Float32Array(maxParticles);
        const opacities = new Float32Array(maxParticles);

        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geo.setAttribute('aScale', new THREE.BufferAttribute(scales, 1));
        geo.setAttribute('aOpacity', new THREE.BufferAttribute(opacities, 1));

        const mat = new THREE.ShaderMaterial({
            uniforms: {
                uColor: { value: new THREE.Color(0x67e8f9) },
                uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) },
            },
            vertexShader: `
                attribute float aScale;
                attribute float aOpacity;
                varying float vOpacity;
                uniform float uPixelRatio;
                void main() {
                    vOpacity = aOpacity;
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    gl_PointSize = aScale * uPixelRatio * 40.0 / -mvPosition.z;
                    gl_Position = projectionMatrix * mvPosition;
                }
            `,
            fragmentShader: `
                uniform vec3 uColor;
                varying float vOpacity;
                void main() {
                    float dist = length(gl_PointCoord - vec2(0.5));
                    if (dist > 0.5) discard;
                    float alpha = smoothstep(0.5, 0.15, dist) * vOpacity;
                    gl_FragColor = vec4(uColor, alpha);
                }
            `,
            transparent: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
        });

        this.particleMesh = new THREE.Points(geo, mat);
        this.particleMesh.frustumCulled = false;
        this.streamlineGroup.add(this.particleMesh);
    }

    clearAll() {
        while (this.tubes.length > 0) {
            const tube = this.tubes.pop();
            this.streamlineGroup.remove(tube);
            if (tube.geometry) tube.geometry.dispose();
            if (tube.material) tube.material.dispose();
        }
        this.curves = [];
        this.particleData = [];
        this._resetParticleBuffers();
    }

    _resetParticleBuffers() {
        const posAttr = this.particleMesh.geometry.getAttribute('position');
        const scaleAttr = this.particleMesh.geometry.getAttribute('aScale');
        const opacAttr = this.particleMesh.geometry.getAttribute('aOpacity');
        for (let i = 0; i < posAttr.count; i++) {
            posAttr.setXYZ(i, 0, -1000, 0);
            scaleAttr.setX(i, 0);
            opacAttr.setX(i, 0);
        }
        posAttr.needsUpdate = true;
        scaleAttr.needsUpdate = true;
        opacAttr.needsUpdate = true;
    }

    addStreamlines(pathlines) {
        this.clearAll();

        const limited = pathlines.slice(0, MAX_STREAMLINES);

        const sharedMat = new THREE.MeshBasicMaterial({
            color: 0x22d3ee,
            transparent: true,
            opacity: 0.45,
            side: THREE.DoubleSide,
        });

        limited.forEach((line, idx) => {
            const pts = line.points || [];
            if (pts.length < 2) return;

            const curvePts = pts.map(p => new THREE.Vector3(p.x, p.y, p.z));
            const curve = new THREE.CatmullRomCurve3(curvePts);
            this.curves.push(curve);

            const tubeGeo = new THREE.TubeGeometry(
                curve, TUBE_SEGMENTS, TUBE_RADIUS, TUBE_RADIAL_SEGMENTS, false
            );

            const nVerts = tubeGeo.attributes.position.count;
            const colors = new Float32Array(nVerts * 3);
            for (let v = 0; v < nVerts; v++) {
                const t = v / (nVerts - 1);
                colors[v * 3] = 0.05 + t * 0.5;
                colors[v * 3 + 1] = 0.75 + t * 0.1;
                colors[v * 3 + 2] = 0.9 - t * 0.1;
            }
            tubeGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

            const tubeMat = new THREE.MeshBasicMaterial({
                vertexColors: true,
                transparent: true,
                opacity: 0.5,
            });

            const mesh = new THREE.Mesh(tubeGeo, tubeMat);
            mesh.userData = { type: 'streamline', lineData: line };
            this.streamlineGroup.add(mesh);
            this.tubes.push(mesh);

            for (let p = 0; p < PARTICLES_PER_LINE; p++) {
                this.particleData.push({
                    curveIndex: this.curves.length - 1,
                    offset: (p / PARTICLES_PER_LINE) + (idx * 0.01),
                    speed: 0.002 + Math.random() * 0.001,
                });
            }
        });

        this._ensureParticleBuffersFit();
    }

    _ensureParticleBuffersFit() {
        const needed = this.particleData.length;
        const current = this.particleMesh.geometry.getAttribute('position').count;
        if (needed > current) {
            const newPos = new Float32Array(needed * 3);
            const newScale = new Float32Array(needed);
            const newOpac = new Float32Array(needed);
            this.particleMesh.geometry.setAttribute('position', new THREE.BufferAttribute(newPos, 3));
            this.particleMesh.geometry.setAttribute('aScale', new THREE.BufferAttribute(newScale, 1));
            this.particleMesh.geometry.setAttribute('aOpacity', new THREE.BufferAttribute(newOpac, 1));
        }
    }

    update(time) {
        if (!this.visible || this.particleData.length === 0) return;

        const posAttr = this.particleMesh.geometry.getAttribute('position');
        const scaleAttr = this.particleMesh.geometry.getAttribute('aScale');
        const opacAttr = this.particleMesh.geometry.getAttribute('aOpacity');

        for (let i = 0; i < this.particleData.length; i++) {
            const pd = this.particleData[i];
            const curve = this.curves[pd.curveIndex];
            if (!curve) continue;

            const pos = (time * pd.speed + pd.offset) % 1;
            const point = curve.getPoint(pos);

            posAttr.setXYZ(i, point.x, point.y, point.z);
            scaleAttr.setX(i, 0.5 + 0.8 * Math.sin(pos * Math.PI));
            opacAttr.setX(i, 0.3 + 0.6 * Math.sin(pos * Math.PI));
        }

        posAttr.needsUpdate = true;
        scaleAttr.needsUpdate = true;
        opacAttr.needsUpdate = true;
    }

    setVisible(v) {
        this.visible = v;
        this.streamlineGroup.visible = v;
    }

    dispose() {
        this.clearAll();
        if (this.particleMesh) {
            if (this.particleMesh.geometry) this.particleMesh.geometry.dispose();
            if (this.particleMesh.material) this.particleMesh.material.dispose();
            this.streamlineGroup.remove(this.particleMesh);
        }
        this.scene.remove(this.streamlineGroup);
    }
}
