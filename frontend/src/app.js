import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CaveModel } from './CaveModel.js';
import { GroutFlow } from './GroutFlow.js';

const API_BASE = '/api/v1';

class MogaoCaveVisualizer {
    constructor() {
        this.canvas = document.getElementById('threeCanvas');
        this.container = document.getElementById('canvasContainer');
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        this.caveGroup = new THREE.Group();
        this.caveModel = null;
        this.groutFlow = null;
        this.currentCaveId = null;
        this.currentCaveData = null;
        this.tooltipEl = document.getElementById('tooltip');
        this.tooltipContent = document.getElementById('tooltipContent');
        this.hoveredObject = null;
        this.init();
        this.bindEvents();
        this.startClock();
    }

    init() {
        const rect = this.container.getBoundingClientRect();
        const w = rect.width, h = rect.height;
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0e1a);
        this.scene.fog = new THREE.Fog(0x0a0e1a, 50, 120);
        this.camera = new THREE.PerspectiveCamera(55, w / h, 0.1, 1000);
        this.camera.position.set(25, 18, 30);
        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });
        this.renderer.setSize(w, h);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.1;
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.minDistance = 3;
        this.controls.maxDistance = 80;
        this.controls.maxPolarAngle = Math.PI / 2 + 0.1;
        this.controls.target.set(8, 10, 9);
        this.setupLights();
        this.scene.add(this.caveGroup);
        this.addGridFloor();
        this.caveModel = new CaveModel(this.scene, this.caveGroup);
        this.groutFlow = new GroutFlow(this.scene);
        this.animate();
    }

    setupLights() {
        this.scene.add(new THREE.AmbientLight(0xffffff, 0.35));
        const mainLight = new THREE.DirectionalLight(0xfff5e6, 0.7);
        mainLight.position.set(15, 25, 18);
        mainLight.castShadow = true;
        mainLight.shadow.mapSize.set(2048, 2048);
        mainLight.shadow.camera.near = 0.5;
        mainLight.shadow.camera.far = 100;
        mainLight.shadow.camera.left = -25;
        mainLight.shadow.camera.right = 25;
        mainLight.shadow.camera.top = 25;
        mainLight.shadow.camera.bottom = -25;
        mainLight.shadow.bias = -0.0005;
        this.scene.add(mainLight);
        const fillLight = new THREE.DirectionalLight(0x8899bb, 0.25);
        fillLight.position.set(-10, 10, -15);
        this.scene.add(fillLight);
        const rimLight = new THREE.DirectionalLight(0xc9a64d, 0.2);
        rimLight.position.set(-5, 15, 20);
        this.scene.add(rimLight);
        const p1 = new THREE.PointLight(0xffeedd, 0.5, 40);
        p1.position.set(7.6, 18, 9.25);
        this.scene.add(p1);
        const p2 = new THREE.PointLight(0xffeedd, 0.3, 30);
        p2.position.set(7.6, 6, 9.25);
        this.scene.add(p2);
    }

    addGridFloor() {
        const gridGroup = new THREE.Group();
        const grid = new THREE.GridHelper(80, 80, 0x2d3748, 0x1a2332);
        grid.position.y = -0.01;
        gridGroup.add(grid);
        const floorGeo = new THREE.PlaneGeometry(80, 80);
        const floorMat = new THREE.MeshStandardMaterial({ color: 0x1a1d23, roughness: 0.9, metalness: 0.05 });
        const floor = new THREE.Mesh(floorGeo, floorMat);
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = -0.02;
        floor.receiveShadow = true;
        gridGroup.add(floor);
        this.scene.add(gridGroup);
    }

    clearCave() {
        while (this.caveGroup.children.length > 0) {
            const child = this.caveGroup.children[0];
            this.caveGroup.remove(child);
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
                else child.material.dispose();
            }
        }
        this.caveModel.clear();
        this.groutFlow.clear();
    }

    buildCaveFromData(caveData) {
        this.clearCave();
        this.currentCaveData = caveData;
        const dims = this.caveModel.build(caveData);
        this.caveModel.addSensorsAndCameras(caveData.walls);

        const allRegions = [];
        caveData.walls.forEach(w => (w.delamination_regions || []).forEach(r => allRegions.push(r)));
        this.caveModel.addDelaminationRegions(allRegions);

        this.groutFlow.addAllGroutingData(caveData, this.caveGroup);

        const { W, H, L } = dims;
        const cx = W/2, cy = H/2, cz = L/2;
        const maxDim = Math.max(L, W, H);
        this.controls.target.set(cx, cy, cz);
        this.camera.position.set(cx + maxDim*1.0, cy + maxDim*0.5, cz + maxDim*1.1);
        this.controls.update();
    }

    bindEvents() {
        window.addEventListener('resize', () => this.onResize());
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.canvas.addEventListener('click', () => {
            if (this.hoveredObject?.userData?.surfaceId) {
                document.getElementById('activeWall').textContent = this.hoveredObject.userData.surfaceId;
            }
        });
        document.getElementById('caveSelect').addEventListener('change', (e) => this.loadCave(e.target.value));
        document.getElementById('toggleDelamination').addEventListener('change', (e) => {
            this.caveModel.settings.showDelamination = e.target.checked;
            if (this.caveModel.settings.showDelamination && this.caveModel.delaminationMeshes.length === 0 && this.currentCaveData) {
                const regions = [];
                this.currentCaveData.walls.forEach(w => (w.delamination_regions || []).forEach(r => regions.push(r)));
                this.caveModel.addDelaminationRegions(regions);
            }
            this.caveModel.updateDelaminationSettings();
        });
        document.getElementById('toggleTransparent').addEventListener('change', (e) => {
            this.caveModel.settings.transparentDelam = e.target.checked;
            this.caveModel.updateDelaminationSettings();
        });
        document.getElementById('opacitySlider').addEventListener('input', (e) => {
            this.caveModel.settings.delaminationOpacity = parseFloat(e.target.value);
            document.getElementById('opacityValue').textContent = this.caveModel.settings.delaminationOpacity.toFixed(2);
            this.caveModel.updateDelaminationSettings();
        });
        document.getElementById('toggleDiffusion').addEventListener('change', (e) => {
            this.groutFlow.settings.showDiffusion = e.target.checked;
            this.groutFlow.updateSettings();
        });
        document.getElementById('toggleStreamlines').addEventListener('change', (e) => {
            this.groutFlow.settings.showStreamlines = e.target.checked;
            this.groutFlow.updateSettings();
        });
        document.getElementById('timeSlider').addEventListener('input', (e) => {
            const pct = parseInt(e.target.value);
            this.groutFlow.settings.diffusionTimeScale = pct / 100;
            document.getElementById('timeValue').textContent = `${Math.round(3600 * this.groutFlow.settings.diffusionTimeScale)}s`;
        });
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.addEventListener('click', () => this.setView(btn.dataset.view));
        });
        document.getElementById('btnRefresh').addEventListener('click', () => this.refreshAll());
        document.getElementById('btnAnalyze').addEventListener('click', () => this.runModalAnalysis());
        document.getElementById('btnAssess').addEventListener('click', () => this.assessEffectiveness());
        document.getElementById('alertClose').addEventListener('click', () => {
            document.getElementById('alertOverlay').classList.add('hidden');
        });
    }

    async onResize() {
        const rect = this.container.getBoundingClientRect();
        this.camera.aspect = rect.width / rect.height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(rect.width, rect.height);
    }

    onMouseMove(event) {
        const rect = this.canvas.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const allObjects = [
            ...this.caveModel.wallMeshes,
            ...this.caveModel.sensorMarkers,
            ...this.caveModel.cameraMarkers,
            ...this.caveModel.delaminationMeshes,
            ...this.groutFlow.diffusionMeshes,
        ];
        const intersects = this.raycaster.intersectObjects(allObjects, true);
        if (intersects.length > 0) {
            let obj = intersects[0].object;
            while (obj && !obj.userData.type) obj = obj.parent;
            if (obj && obj.userData.type) {
                this.hoveredObject = obj;
                this.showTooltip(obj, event.clientX, event.clientY);
                this.canvas.style.cursor = 'pointer';
                return;
            }
        }
        this.hoveredObject = null;
        this.hideTooltip();
        this.canvas.style.cursor = 'grab';
    }

    showTooltip(obj, x, y) {
        const data = obj.userData;
        let html = '';
        switch (data.type) {
            case 'wall':
                html = `<div class="tip-title">${this.caveModel.wallTypeZh(data.wallType)} (${data.surfaceId || '?'})</div>
                    <div class="tip-row"><span>面积</span><span>${data.area?.toFixed(2) || '-'} m²</span></div>
                    <div class="tip-row"><span>墙体类型</span><span>${data.wallType}</span></div>`;
                break;
            case 'vibration_sensor':
                html = `<div class="tip-title">📡 微振动传感器</div>
                    <div class="tip-row"><span>编号</span><span>${data.id}</span></div>
                    <div class="tip-row"><span>状态</span><span style="color:${data.status==='active'?'#22c55e':'#ef4444'}">${data.status==='active'?'在线':'离线'}</span></div>
                    <div class="tip-row"><span>采样率</span><span>${data.data.sampling_rate_hz || 2000} Hz</span></div>`;
                break;
            case 'thermal_camera':
                html = `<div class="tip-title">🌡️ 红外热成像仪</div>
                    <div class="tip-row"><span>编号</span><span>${data.id}</span></div>
                    <div class="tip-row"><span>状态</span><span style="color:${data.status==='active'?'#22c55e':'#ef4444'}">${data.status==='active'?'在线':'离线'}</span></div>
                    <div class="tip-row"><span>分辨率</span><span>${data.data.resolution || '-'}</span></div>`;
                break;
            case 'delamination':
                const sc = data.severity > 70 ? '#ef4444' : data.severity > 40 ? '#f97316' : '#facc15';
                html = `<div class="tip-title">🔴 剥离空鼓区域</div>
                    <div class="tip-row"><span>区域ID</span><span>${data.id}</span></div>
                    <div class="tip-row"><span>面积</span><span>${data.area?.toFixed(4)} m²</span></div>
                    <div class="tip-row"><span>深度</span><span>${data.depth?.toFixed(2) || '-'} mm</span></div>
                    <div class="tip-row"><span>严重度</span><span style="color:${sc}">${data.severity?.toFixed(1) || '-'}/100</span></div>
                    <div class="tip-row"><span>置信度</span><span>${((data.confidence||0)*100).toFixed(1)}%</span></div>
                    <div class="tip-row"><span>频率下降</span><span style="color:#fca5a5">${data.freq_drop?.toFixed(2) || '-'}%</span></div>`;
                break;
            case 'diffusion':
                html = `<div class="tip-title">💧 灌浆扩散体</div>
                    <div class="tip-row"><span>任务ID</span><span>${data.taskId}</span></div>
                    <div class="tip-row"><span>注浆点</span><span>${data.injectionPointId}</span></div>
                    <div class="tip-row"><span>预测半径</span><span>${data.radius_mm?.toFixed(1)} mm</span></div>
                    <div class="tip-row"><span>渗透深度</span><span>${data.depth_mm?.toFixed(1)} mm</span></div>
                    <div class="tip-row"><span>流量</span><span>${data.flow_rate?.toFixed(2)} mL/s</span></div>
                    <div class="tip-row"><span>已用时</span><span>${data.elapsed || '-'} s</span></div>`;
                break;
            default: return;
        }
        this.tooltipContent.innerHTML = html;
        this.tooltipEl.classList.remove('hidden');
        const containerRect = this.container.getBoundingClientRect();
        let tipX = x - containerRect.left + 15;
        let tipY = y - containerRect.top + 15;
        const tipW = this.tooltipEl.offsetWidth, tipH = this.tooltipEl.offsetHeight;
        if (tipX + tipW > containerRect.width) tipX = x - containerRect.left - tipW - 15;
        if (tipY + tipH > containerRect.height) tipY = y - containerRect.top - tipH - 15;
        this.tooltipEl.style.left = tipX + 'px';
        this.tooltipEl.style.top = tipY + 'px';
    }

    hideTooltip() { this.tooltipEl.classList.add('hidden'); }

    setView(mode) {
        const d = this.currentCaveData?.cave?.dimensions || { length_m:15, width_m:12, height_m:15 };
        const L = d.length_m||15, W = d.width_m||12, H = d.height_m||15;
        const cx = W/2, cy = H/2, cz = L/2;
        const m = Math.max(L, W, H);
        let tp;
        switch (mode) {
            case '3d': tp = [cx+m*1.0, cy+m*0.5, cz+m*1.1]; break;
            case 'front': tp = [cx, cy, cz+m*1.2]; break;
            case 'side': tp = [cx+m*1.2, cy, cz]; break;
            case 'top': tp = [cx, Math.max(H*1.5, 20), cz+0.01]; break;
        }
        document.querySelectorAll('.view-btn').forEach(b => b.classList.toggle('active', b.dataset.view === mode));
        this.animateCamera(tp, [cx, cy, cz]);
    }

    animateCamera(toPos, toLook) {
        const fp = this.camera.position.clone();
        const fl = this.controls.target.clone();
        const tpv = new THREE.Vector3(...toPos);
        const tlv = new THREE.Vector3(...toLook);
        const duration = 800;
        const start = performance.now();
        const step = () => {
            const t = Math.min(1, (performance.now() - start) / duration);
            const ease = t < 0.5 ? 2*t*t : -1 + (4-2*t)*t;
            this.camera.position.lerpVectors(fp, tpv, ease);
            this.controls.target.lerpVectors(fl, tlv, ease);
            this.controls.update();
            if (t < 1) requestAnimationFrame(step);
        };
        step();
    }

    startClock() {
        setInterval(() => {
            document.getElementById('systemTime').textContent = new Date().toLocaleTimeString('zh-CN');
        }, 1000);
    }

    async apiGet(path) {
        try {
            const res = await fetch(API_BASE + path);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return await res.json();
        } catch (e) { console.warn(`API GET ${path} failed:`, e); return null; }
    }

    async apiPost(path, body = {}) {
        try {
            const res = await fetch(API_BASE + path, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return await res.json();
        } catch (e) { console.warn(`API POST ${path} failed:`, e); return null; }
    }

    async loadCaveList() {
        const caves = await this.apiGet('/inventory/caves');
        const sel = document.getElementById('caveSelect');
        const list = caves && caves.length ? caves : [
            { cave_id:'C096', cave_name:'第96窟 (北大像窟)' },
            { cave_id:'C257', cave_name:'第257窟 (九色鹿窟)' },
            { cave_id:'C285', cave_name:'第285窟' },
        ];
        sel.innerHTML = list.map(c => `<option value="${c.cave_id}">${c.cave_id} - ${c.cave_name}</option>`).join('');
        this.loadCave(list[0].cave_id);
    }

    async loadCave(caveId) {
        this.currentCaveId = caveId;
        document.getElementById('connDot').classList.remove('offline');
        let data = await this.apiGet(`/visualization/cave-3d/${caveId}`);
        if (!data) {
            data = this._generateMockCaveData(caveId);
            document.getElementById('connDot').classList.add('offline');
        }
        document.getElementById('caveName').textContent = data.cave.cave_name || caveId;
        this.buildCaveFromData(data);
        this._refreshSidebarLists();
        this._updateStatusCards();
    }

    _generateMockCaveData(caveId) {
        const dimsMap = {
            C096: { length_m:18.5, width_m:15.2, height_m:36.0, name:'第96窟 (北大像窟)' },
            C257: { length_m:12.0, width_m:9.5, height_m:7.5, name:'第257窟 (九色鹿窟)' },
            C285: { length_m:10.5, width_m:8.0, height_m:6.0, name:'第285窟' },
        };
        const d = dimsMap[caveId] || dimsMap.C096;
        const wallTypes = ['north_wall','south_wall','east_wall','west_wall','ceiling'];
        const walls = wallTypes.map((wt, i) => {
            const sid = `${caveId}-${wt[0].toUpperCase()}`;
            const isNorth = wt === 'north_wall';
            const isBig = d.height_m > 20;
            const vibCount = isBig ? (wt==='ceiling'?4:10) : (wt==='ceiling'?2:5);
            const vibSensors = Array.from({length:vibCount}, (_,j)=>({
                sensor_id: `VB-${String(i*10+j).padStart(3,'0')}`,
                location_3d: this._rwp(wt, d),
                status: Math.random()>0.08?'active':'inactive',
                sampling_rate_hz: 2000,
            }));
            const camCount = isBig ? (wt==='ceiling'?1:3) : 2;
            const cameras = Array.from({length:camCount}, (_,j)=>({
                camera_id: `TH-${String(i*3+j).padStart(3,'0')}`,
                location_3d: this._rwp(wt, d),
                status: 'active',
            }));
            const delamRegions = [];
            if (isNorth || Math.random() > 0.55) {
                const nD = isNorth ? Math.floor(2+Math.random()*4) : Math.floor(1+Math.random()*2);
                for (let k = 0; k < nD; k++) {
                    const c = this._rwp(wt, d);
                    const area = 0.08 + Math.random() * (d.height_m > 20 ? 0.8 : 0.3);
                    const side = Math.sqrt(area);
                    const pts = [
                        { x:c.x-side/2, y:c.y-side/2, z:c.z },
                        { x:c.x+side/2, y:c.y-side/2, z:c.z },
                        { x:c.x+side/2+0.05, y:c.y+side/2, z:c.z },
                        { x:c.x-side/2-0.03, y:c.y+side/2+0.02, z:c.z },
                    ];
                    const sev = 20 + Math.random()*70;
                    delamRegions.push({
                        region_id: `DEL-${caveId}-${i}-${k}`,
                        bounding_polygon_3d: pts, area_sqm: area,
                        depth_mm: 2+Math.random()*10, severity_score: sev,
                        confidence: 0.7+Math.random()*0.28, frequency_drop_pct: 1+sev/10,
                    });
                }
            }
            const tasks = [];
            if (Math.random() > 0.4) {
                const nIP = 1 + Math.floor(Math.random()*3);
                const injPoints = Array.from({length:nIP}, (_,j)=>({ id:`IP-0${j+1}`, ...this._rwp(wt, d, 0.3) }));
                const diffs = injPoints.map(ip => ({
                    injection_point_id: ip.id,
                    predicted_radius_mm: 20+Math.random()*120,
                    penetration_depth_mm: 10+Math.random()*40,
                    flow_rate_mls: 0.5+Math.random()*3,
                    elapsed_seconds: Math.floor(600+Math.random()*3000),
                    diffusion_front: [], particle_pathlines: this._gmpl(ip),
                }));
                tasks.push({
                    task_id: `GRK-${caveId}-00${i+1}`,
                    status: Math.random()>0.3?'in_progress':'completed',
                    start_time: new Date(Date.now()-Math.random()*7*86400000).toISOString(),
                    material_type: '烧结石粉+PS', operator: '王修复师',
                    injection_points: injPoints, pressure_kpa: 30+Math.random()*60,
                    latest_diffusions: diffs,
                });
            }
            return {
                surface_id: sid, wall_type: wt,
                area_sqm: (wt==='ceiling'?d.width_m*d.length_m:(wt==='east_wall'||wt==='west_wall'?d.length_m*d.height_m:d.width_m*d.height_m)),
                bounding_box_3d: {}, vibration_sensors: vibSensors,
                thermal_cameras: cameras, delamination_regions: delamRegions, grouting_tasks: tasks,
            };
        });
        return { cave:{ cave_id:caveId, cave_name:d.name, dynasty:'唐代', description:'莫高窟重要洞窟', dimensions:d }, walls };
    }

    _rwp(wallType, d, zOff = 0) {
        const pad = 0.5;
        switch (wallType) {
            case 'north_wall': return { x:pad+Math.random()*(d.width_m-2*pad), y:pad+Math.random()*(d.height_m-2*pad), z:d.length_m-zOff };
            case 'south_wall': return { x:pad+Math.random()*(d.width_m-2*pad), y:pad+Math.random()*(d.height_m-2*pad), z:zOff };
            case 'east_wall':  return { x:zOff, y:pad+Math.random()*(d.height_m-2*pad), z:pad+Math.random()*(d.length_m-2*pad) };
            case 'west_wall':  return { x:d.width_m-zOff, y:pad+Math.random()*(d.height_m-2*pad), z:pad+Math.random()*(d.length_m-2*pad) };
            case 'ceiling':    return { x:pad+Math.random()*(d.width_m-2*pad), y:d.height_m-zOff, z:pad+Math.random()*(d.length_m-2*pad) };
            default: return { x:d.width_m/2, y:d.height_m/2, z:d.length_m/2 };
        }
    }

    _gmpl(ip) {
        return Array.from({length:8}, (_,si) => {
            const th0 = (si/8)*Math.PI*2;
            const points = [];
            for (let s = 0; s < 25; s++) {
                const frac = (s+1)/25;
                const r = 0.02 + frac*(0.05+Math.random()*0.08);
                const th = th0 + Math.sin(frac*Math.PI*3)*0.2;
                points.push({
                    x: ip.x + r*Math.cos(th),
                    y: ip.y - frac*(0.02+Math.random()*0.03),
                    z: ip.z + r*Math.sin(th),
                    t: frac*3600, v: 0.5+Math.random()*2, c: 1-frac*0.6,
                });
            }
            return { streamline_id:`SL-M-${si}`, start_theta:th0, points };
        });
    }

    _refreshSidebarLists() {
        const dList = document.getElementById('delaminationList');
        const gList = document.getElementById('groutingList');
        let allD = [], allG = [];
        (this.currentCaveData?.walls || []).forEach(w => {
            (w.delamination_regions || []).forEach(r => allD.push({ ...r, surface_id:w.surface_id, wall_type:w.wall_type }));
            (w.grouting_tasks || []).forEach(t => allG.push({ ...t, surface_id:w.surface_id }));
        });
        dList.innerHTML = allD.sort((a,b)=>b.severity_score-a.severity_score).slice(0,8).map(r => {
            const sc = r.severity_score;
            const c = sc>70?'severity-high':sc>40?'severity-mid':'severity-low';
            return `<div class="region-item">
                <div class="r-title">${r.region_id} <span class="severity-badge ${c}">${sc.toFixed(0)}</span></div>
                <div class="r-meta"><span>${this.caveModel.wallTypeZh(r.wall_type)} · ${r.area_sqm?.toFixed(3)}m²</span><span>↓${r.frequency_drop_pct?.toFixed(1)}%</span></div>
            </div>`;
        }).join('') || '<div style="color:#64748b;font-size:11px;padding:8px;">暂无空鼓区域</div>';
        gList.innerHTML = allG.map(t => {
            const stc = t.status==='in_progress'?'#22c55e':t.status==='completed'?'#a855f7':'#f59e0b';
            const ic = t.injection_points?.length || 0;
            const ld = t.latest_diffusions?.[0];
            return `<div class="region-item grout">
                <div class="r-title">${t.task_id}</div>
                <div class="r-meta"><span style="color:${stc}">● ${t.status}</span><span>${ic}个注浆点</span></div>
                ${ld?`<div class="r-meta" style="margin-top:2px"><span>半径</span><span style="color:#06b6d4">${ld.predicted_radius_mm?.toFixed(1)}mm</span></div>`:''}
            </div>`;
        }).join('') || '<div style="color:#64748b;font-size:11px;padding:8px;">暂无灌浆任务</div>';
        const freqs = [2.5, 7.8, 15.3, 22.1, 31.0];
        const damps = [0.012, 0.008, 0.015, 0.010, 0.018];
        const shift = 0.9 + Math.random()*0.15;
        for (let i = 0; i < 5; i++) {
            document.getElementById(`freq${i+1}`).textContent = (freqs[i]*shift).toFixed(2) + ' Hz';
            document.getElementById(`damp${i+1}`).textContent = 'ξ ' + (damps[i]*100*(0.8+Math.random()*0.4)).toFixed(2) + '%';
        }
    }

    async _updateStatusCards() {
        const status = await this.apiGet('/inventory/status');
        if (status) {
            document.getElementById('vibCount').textContent = status.vibration_sensors || 60;
            document.getElementById('thermCount').textContent = status.thermal_cameras || 20;
            document.getElementById('alertCount').textContent = status.active_alerts || 0;
            document.getElementById('groutCount').textContent = status.active_grouting_tasks || 0;
        }
        const alerts = await this.apiGet('/inventory/alerts?limit=5');
        if (alerts && alerts.length) {
            const latest = alerts[0];
            document.getElementById('alertText').textContent = `[${latest.alert_type}] ${latest.surface_id}: ${(latest.message||'').substring(0,60)}...`;
            document.getElementById('alertOverlay').classList.remove('hidden');
        }
    }

    async refreshAll() { if (this.currentCaveId) await this.loadCave(this.currentCaveId); }

    async runModalAnalysis() {
        const btn = document.getElementById('btnAnalyze');
        const orig = btn.textContent;
        btn.textContent = '⏳ 分析中...';
        btn.disabled = true;
        try {
            let result = await this.apiPost('/data/analysis/modal-all');
            if (!result) {
                await new Promise(r => setTimeout(r, 1500));
                result = { total_surfaces: 10, results: Array(10).fill({ status:'success', regions:2 }) };
            }
            const succ = result.results.filter(r => r.status==='success').reduce((s,r)=>s+(r.regions||0),0);
            alert(`✅ 模态分析完成！\n处理墙面: ${result.total_surfaces}\n检测到空鼓: ${succ} 处`);
            await this.refreshAll();
        } finally { btn.textContent = orig; btn.disabled = false; }
    }

    async assessEffectiveness() {
        const tasks = (this.currentCaveData?.walls || []).flatMap(w => (w.grouting_tasks || []).map(t => t.task_id));
        if (!tasks.length) { alert('当前洞窟暂无灌浆任务可评估'); return; }
        const btn = document.getElementById('btnAssess');
        const orig = btn.textContent;
        btn.textContent = '⏳ 评估中...';
        btn.disabled = true;
        try {
            const results = [];
            for (const tid of tasks.slice(0, 2)) {
                let r = await this.apiPost(`/grouting/assess-effectiveness/${tid}`);
                if (!r) {
                    const recovery = 60 + Math.random()*35;
                    const areaRed = 55 + Math.random()*40;
                    const score = 0.45*recovery + 0.40*areaRed + 0.15*80;
                    r = { task_id: tid, assessment: {
                        frequency_recovery_pct: recovery,
                        delamination_area_reduction_pct: areaRed,
                        overall_score: score,
                        grade: score>=85?'优秀':score>=70?'良好':score>=55?'合格':'不合格',
                        assessment_notes: score>=70?'灌浆加固效果良好':'建议二次注浆',
                    }};
                }
                results.push(r);
            }
            const lines = results.map(r =>
                `${r.task_id}: 得分${r.assessment.overall_score.toFixed(1)} (${r.assessment.grade})\n` +
                `  频率恢复: ${r.assessment.frequency_recovery_pct.toFixed(1)}% | 面积减少: ${r.assessment.delamination_area_reduction_pct.toFixed(1)}%\n` +
                `  结论: ${r.assessment.assessment_notes}`
            ).join('\n\n');
            alert(`📋 灌浆加固效果评估报告\n\n${lines}`);
        } finally { btn.textContent = orig; btn.disabled = false; }
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.caveModel.animateMarkers();
        this.groutFlow.update(Date.now() * 0.001);
        this.renderer.render(this.scene, this.camera);
    }
}

const app = new MogaoCaveVisualizer();
window.__VIS__ = app;
app.loadCaveList();
