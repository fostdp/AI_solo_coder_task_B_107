import * as THREE from 'three';

export class CaveModel {
    constructor(scene, caveGroup) {
        this.scene = scene;
        this.caveGroup = caveGroup;
        this.wallMeshes = [];
        this.sensorMarkers = [];
        this.cameraMarkers = [];
        this.delaminationMeshes = [];
        this.animationIds = new Map();
        this.currentCaveData = null;
        this.settings = {
            showDelamination: true,
            delaminationOpacity: 0.45,
            transparentDelam: true,
        };
        this.wallColors = {
            north_wall: 0x8B7355, south_wall: 0x8B7355,
            east_wall: 0x806858, west_wall: 0x806858,
            ceiling: 0x9B8B75, floor: 0x6B5B4F,
        };
    }

    clear() {
        this.wallMeshes = [];
        this.sensorMarkers = [];
        this.cameraMarkers = [];
        this.clearDelamination();
    }

    clearDelamination() {
        this.delaminationMeshes.forEach(m => {
            this.caveGroup.remove(m);
            if (m.geometry) m.geometry.dispose();
            if (m.material) {
                if (Array.isArray(m.material)) m.material.forEach(mm => mm.dispose());
                else m.material.dispose();
            }
        });
        this.delaminationMeshes = [];
        this.animationIds.forEach(id => cancelAnimationFrame(id));
        this.animationIds.clear();
    }

    build(caveData) {
        this.currentCaveData = caveData;
        const dims = caveData.cave.dimensions || { length_m: 15, width_m: 12, height_m: 15 };
        const L = dims.length_m || 15, W = dims.width_m || 12, H = dims.height_m || 15;

        const walls = [
            { type: 'north_wall', w: W, h: H, pos: [W/2, H/2, L], rot: [0, 0, 0] },
            { type: 'south_wall', w: W, h: H, pos: [W/2, H/2, 0], rot: [0, Math.PI, 0] },
            { type: 'east_wall',  w: L, h: H, pos: [0, H/2, L/2], rot: [0, Math.PI/2, 0] },
            { type: 'west_wall',  w: L, h: H, pos: [W, H/2, L/2], rot: [0, -Math.PI/2, 0] },
            { type: 'ceiling',    w: W, h: L, pos: [W/2, H, L/2], rot: [-Math.PI/2, 0, 0] },
        ];
        const typeToSurfaceMap = {};
        caveData.walls.forEach(w => { typeToSurfaceMap[w.wall_type] = w; });
        let totalArea = 0;

        walls.forEach(w => {
            const surfaceData = typeToSurfaceMap[w.type];
            const color = this.wallColors[w.type] || 0x8B7355;
            const geo = new THREE.PlaneGeometry(w.w, w.h, Math.round(w.w), Math.round(w.h));
            const positions = geo.attributes.position;
            for (let i = 0; i < positions.count; i++) {
                const nx = positions.getX(i), ny = positions.getY(i);
                positions.setZ(i, (Math.random()-0.5)*0.02 + Math.sin(nx*2)*0.008 + Math.cos(ny*1.5)*0.006);
            }
            geo.computeVertexNormals();
            const texCanvas = this._createWallTexture(w.type, color);
            const texture = new THREE.CanvasTexture(texCanvas);
            texture.wrapS = THREE.RepeatWrapping;
            texture.wrapT = THREE.RepeatWrapping;
            texture.repeat.set(Math.max(1, Math.round(w.w/2)), Math.max(1, Math.round(w.h/2)));
            const mat = new THREE.MeshStandardMaterial({
                map: texture, color: 0xffffff, roughness: 0.92, metalness: 0.03, side: THREE.DoubleSide,
            });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.set(w.pos[0], w.pos[1], w.pos[2]);
            mesh.rotation.set(w.rot[0], w.rot[1], w.rot[2]);
            mesh.receiveShadow = true;
            mesh.castShadow = true;
            mesh.userData = {
                type: 'wall', wallType: w.type, surfaceId: surfaceData?.surface_id,
                area: w.w * w.h, data: surfaceData || {},
            };
            const edgesGeo = new THREE.EdgesGeometry(geo);
            const edgeMat = new THREE.LineBasicMaterial({ color: 0x3d2817, transparent: true, opacity: 0.5 });
            mesh.add(new THREE.LineSegments(edgesGeo, edgeMat));
            this.caveGroup.add(mesh);
            this.wallMeshes.push(mesh);
            totalArea += w.w * w.h;
        });

        this._addFrameStructure(W, H, L);
        document.getElementById('caveDims').textContent = `${L} × ${W} × ${H} m`;
        document.getElementById('caveArea').textContent = `${totalArea.toFixed(1)} m²`;
        return { W, H, L };
    }

    addSensorsAndCameras(walls) {
        walls.forEach(wall => {
            (wall.vibration_sensors || []).forEach(s => this._addVibrationSensor(s));
            (wall.thermal_cameras || []).forEach(c => this._addThermalCamera(c));
        });
    }

    addDelaminationRegions(regions) {
        regions.forEach(r => this._addDelaminationRegion(r));
    }

    updateDelaminationSettings() {
        this.delaminationMeshes.forEach(mesh => {
            mesh.material.opacity = this.settings.transparentDelam ? this.settings.delaminationOpacity : 0.95;
            mesh.material.transparent = this.settings.transparentDelam;
            mesh.visible = this.settings.showDelamination;
        });
    }

    animateMarkers() {
        this.sensorMarkers.forEach((s, i) => {
            s.rotation.y += 0.005;
            s.children[2] && (s.children[2].position.y = 0.16 + Math.sin(Date.now()*0.003 + i)*0.008);
        });
        this.cameraMarkers.forEach((c, i) => {
            const flash = Math.sin(Date.now()*0.004 + i) > 0.7;
            c.children[1] && (c.children[1].material.emissiveIntensity = flash ? 1.2 : 0.4);
        });
    }

    guessWallType(z) {
        const dims = this.currentCaveData?.cave?.dimensions || {};
        const L = dims.length_m || 15, W = dims.width_m || 12;
        if (Math.abs(z - L) < 0.3) return 'north_wall';
        if (Math.abs(z) < 0.3) return 'south_wall';
        if (Math.abs(z - W) < 0.3) return 'west_wall';
        return 'east_wall';
    }

    wallTypeZh(type) {
        return { north_wall:'北墙', south_wall:'南墙', east_wall:'东墙', west_wall:'西墙', ceiling:'窟顶', floor:'地面' }[type] || type;
    }

    _addFrameStructure(W, H, L) {
        const frameMat = new THREE.MeshStandardMaterial({ color: 0x4a3728, roughness: 0.85, metalness: 0.05 });
        const pillarR = 0.12;
        const corners = [[0,0,0],[W,0,0],[0,0,L],[W,0,L]];
        corners.forEach(c => {
            const geo = new THREE.CylinderGeometry(pillarR, pillarR*1.1, H, 12);
            const mesh = new THREE.Mesh(geo, frameMat);
            mesh.position.set(c[0]+(c[0]===0?pillarR:-pillarR), H/2, c[2]+(c[2]===0?pillarR:-pillarR));
            mesh.castShadow = true; mesh.receiveShadow = true;
            this.caveGroup.add(mesh);
        });
        const beamGeo = new THREE.BoxGeometry(W+0.24, 0.2, 0.2);
        const b1 = new THREE.Mesh(beamGeo, frameMat); b1.position.set(W/2, H, 0); this.caveGroup.add(b1);
        const b2 = new THREE.Mesh(beamGeo, frameMat); b2.position.set(W/2, H, L); this.caveGroup.add(b2);
        const beamGeo2 = new THREE.BoxGeometry(0.2, 0.2, L+0.24);
        const b3 = new THREE.Mesh(beamGeo2, frameMat); b3.position.set(0, H, L/2); this.caveGroup.add(b3);
        const b4 = new THREE.Mesh(beamGeo2, frameMat); b4.position.set(W, H, L/2); this.caveGroup.add(b4);
    }

    _createWallTexture(type, baseColor) {
        const c = document.createElement('canvas');
        c.width = 256; c.height = 256;
        const ctx = c.getContext('2d');
        const hex = baseColor.toString(16).padStart(6, '0');
        ctx.fillStyle = `#${hex}`;
        ctx.fillRect(0, 0, 256, 256);
        for (let i = 0; i < 6000; i++) {
            const x = Math.random()*256, y = Math.random()*256, a = Math.random()*0.08;
            const shade = Math.random() > 0.5 ? 0 : 255;
            ctx.fillStyle = `rgba(${shade},${shade},${shade},${a})`;
            ctx.fillRect(x, y, 1+Math.random()*2, 1+Math.random()*2);
        }
        for (let i = 0; i < 15; i++) {
            const x = Math.random()*256, y = Math.random()*256, w = 20+Math.random()*80;
            ctx.strokeStyle = `rgba(40,25,15,${0.05+Math.random()*0.08})`;
            ctx.lineWidth = 0.5 + Math.random();
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.bezierCurveTo(x+w*0.3, y+(Math.random()-0.5)*15, x+w*0.7, y+(Math.random()-0.5)*15, x+w, y+(Math.random()-0.5)*8);
            ctx.stroke();
        }
        const fgColors = [
            { color:'rgba(180,60,50,0.06)', x:0.2, y:0.3, r:0.3 },
            { color:'rgba(60,100,150,0.05)', x:0.7, y:0.6, r:0.25 },
            { color:'rgba(160,120,50,0.05)', x:0.5, y:0.8, r:0.28 },
        ];
        fgColors.forEach(fg => {
            const grd = ctx.createRadialGradient(fg.x*256, fg.y*256, 10, fg.x*256, fg.y*256, fg.r*256);
            grd.addColorStop(0, fg.color);
            grd.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = grd;
            ctx.fillRect(0, 0, 256, 256);
        });
        return c;
    }

    _addVibrationSensor(sensor) {
        const loc = sensor.location_3d;
        if (!loc) return;
        const group = new THREE.Group();
        const baseGeo = new THREE.CylinderGeometry(0.08, 0.1, 0.04, 16);
        const baseMat = new THREE.MeshStandardMaterial({ color:0x3b82f6, roughness:0.4, metalness:0.6, emissive:0x1e3a8a, emissiveIntensity:0.2 });
        group.add(new THREE.Mesh(baseGeo, baseMat));
        const bodyGeo = new THREE.CylinderGeometry(0.05, 0.065, 0.12, 16);
        const bodyMat = new THREE.MeshStandardMaterial({ color:0x60a5fa, roughness:0.3, metalness:0.7, emissive:0x3b82f6, emissiveIntensity:0.15 });
        const body = new THREE.Mesh(bodyGeo, bodyMat); body.position.y = 0.08; group.add(body);
        const topGeo = new THREE.SphereGeometry(0.035, 16, 16);
        const topMat = new THREE.MeshStandardMaterial({ color:0xfbbf24, roughness:0.2, metalness:0.8, emissive:0xfbbf24, emissiveIntensity:0.4 });
        const top = new THREE.Mesh(topGeo, topMat); top.position.y = 0.16; group.add(top);
        group.position.set(loc.x, loc.y, loc.z);
        group.castShadow = true;
        group.userData = { type: 'vibration_sensor', id: sensor.sensor_id, status: sensor.status, data: sensor };
        this.caveGroup.add(group);
        this.sensorMarkers.push(group);
    }

    _addThermalCamera(camera) {
        const loc = camera.location_3d;
        if (!loc) return;
        const group = new THREE.Group();
        const bodyGeo = new THREE.BoxGeometry(0.12, 0.08, 0.18);
        const bodyMat = new THREE.MeshStandardMaterial({ color:0x374151, roughness:0.5, metalness:0.5, emissive:0x7c2d12, emissiveIntensity:0.15 });
        group.add(new THREE.Mesh(bodyGeo, bodyMat));
        const lensGeo = new THREE.CylinderGeometry(0.045, 0.045, 0.06, 18);
        const lensMat = new THREE.MeshStandardMaterial({ color:0x0ea5e9, roughness:0.1, metalness:0.9, emissive:0x0284c7, emissiveIntensity:0.5, transparent:true, opacity:0.85 });
        const lens = new THREE.Mesh(lensGeo, lensMat); lens.rotation.x = Math.PI/2; lens.position.z = 0.12; group.add(lens);
        const mountGeo = new THREE.CylinderGeometry(0.03, 0.05, 0.08, 12);
        const mountMat = new THREE.MeshStandardMaterial({ color:0x4b5563, roughness:0.7, metalness:0.3 });
        const mount = new THREE.Mesh(mountGeo, mountMat); mount.position.z = -0.07; group.add(mount);
        group.position.set(loc.x, loc.y, loc.z);
        group.rotation.y = Math.PI;
        group.castShadow = true;
        group.userData = { type: 'thermal_camera', id: camera.camera_id, status: camera.status, data: camera };
        this.caveGroup.add(group);
        this.cameraMarkers.push(group);
    }

    _addDelaminationRegion(region) {
        if (!this.settings.showDelamination) return;
        const poly = region.bounding_polygon_3d;
        if (!poly || poly.length < 3) return;
        try {
            const shape = new THREE.Shape();
            const pts2d = poly.map(p => new THREE.Vector2(p.x, p.y));
            shape.moveTo(pts2d[0].x, pts2d[0].y);
            for (let i = 1; i < pts2d.length; i++) shape.lineTo(pts2d[i].x, pts2d[i].y);
            shape.closePath();
            const depth = Math.max(0.005, (region.depth_mm || 5) / 1000);
            const geo = new THREE.ExtrudeGeometry(shape, {
                depth: depth, bevelEnabled: true, bevelThickness: depth*0.3,
                bevelSize: depth*0.3, bevelSegments: 2, steps: 1,
            });
            const severity = region.severity_score || 50;
            const sn = Math.min(1, severity / 100);
            const color = new THREE.Color(0.95, 0.1 + 0.25*(1-sn), 0.1 + 0.1*(1-sn));
            const opacity = this.settings.transparentDelam ? this.settings.delaminationOpacity : 0.95;
            const mat = new THREE.MeshPhysicalMaterial({
                color: color, transparent: this.settings.transparentDelam, opacity: opacity,
                roughness: 0.3, metalness: 0.1, emissive: color, emissiveIntensity: 0.35 * sn,
                side: THREE.DoubleSide, clearcoat: 0.3, clearcoatRoughness: 0.4,
            });
            const mesh = new THREE.Mesh(geo, mat);
            const refZ = poly[0].z;
            const wallType = this.guessWallType(refZ);
            if (wallType === 'south_wall') { mesh.position.z = refZ - depth; }
            else if (wallType === 'north_wall') { mesh.position.z = refZ; }
            else if (wallType === 'east_wall') { mesh.rotation.y = -Math.PI/2; mesh.position.x = refZ; }
            else if (wallType === 'west_wall') { mesh.rotation.y = Math.PI/2; mesh.position.x = refZ; }
            mesh.castShadow = true;
            mesh.userData = {
                type: 'delamination', id: region.region_id, area: region.area_sqm,
                depth: region.depth_mm, severity: region.severity_score,
                confidence: region.confidence, freq_drop: region.frequency_drop_pct, data: region,
            };
            const edges = new THREE.EdgesGeometry(geo);
            const edgeMat = new THREE.LineBasicMaterial({ color: 0xff3333, transparent: true, opacity: 0.8 });
            mesh.add(new THREE.LineSegments(edges, edgeMat));
            this.caveGroup.add(mesh);
            this.delaminationMeshes.push(mesh);
            this._animatePulse(mesh, sn);
        } catch (e) { console.warn('Delamination mesh error:', e); }
    }

    _animatePulse(mesh, intensity) {
        let phase = Math.random() * Math.PI * 2;
        const baseEmissive = mesh.material.emissiveIntensity;
        const tick = () => {
            phase += 0.03;
            const pulse = 0.6 + 0.4 * Math.sin(phase) * intensity;
            mesh.material.emissiveIntensity = baseEmissive * pulse;
            mesh.material.opacity = this.settings.delaminationOpacity * (0.85 + 0.15 * Math.sin(phase*0.7));
            this.animationIds.set(`p_${mesh.uuid}`, requestAnimationFrame(tick));
        };
        tick();
    }
}
