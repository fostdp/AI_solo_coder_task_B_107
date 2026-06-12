import * as THREE from 'three';
import { FlowLineManager } from './flow_lines.js';

export class GroutFlow {
    constructor(scene) {
        this.scene = scene;
        this.diffusionMeshes = [];
        this.flowLineMgr = new FlowLineManager(scene);
        this.settings = {
            showDiffusion: true,
            showStreamlines: true,
            diffusionTimeScale: 1.0,
        };
    }

    clear() {
        this.diffusionMeshes.forEach(m => {
            if (m.parent) m.parent.remove(m);
            if (m.geometry) m.geometry.dispose();
            if (m.material) {
                if (Array.isArray(m.material)) m.material.forEach(mm => mm.dispose());
                else m.material.dispose();
            }
        });
        this.diffusionMeshes = [];
        this.flowLineMgr.clearAll();
    }

    addDiffusionSphere(taskId, ip, diff) {
        if (!this.settings.showDiffusion) return;
        const r_mm = diff.predicted_radius_mm || 50;
        const r = r_mm / 1000 * this.settings.diffusionTimeScale;
        const depth_mm = diff.penetration_depth_mm || 30;
        const depth = depth_mm / 1000;
        const scale = 0.8 + Math.random() * 0.4;
        const geo = new THREE.SphereGeometry(Math.max(0.02, r * scale), 48, 36);
        const positions = geo.attributes.position;
        for (let i = 0; i < positions.count; i++) {
            const y = positions.getY(i);
            const ys = 0.3 + 0.7 * (depth / Math.max(r, 0.001));
            positions.setY(i, y * ys);
            positions.setZ(i, positions.getZ(i) * (0.9 + 0.2*Math.random()));
        }
        geo.computeVertexNormals();
        const mat = new THREE.MeshPhysicalMaterial({
            color: 0x06b6d4, transparent: true, opacity: 0.25, roughness: 0.1,
            metalness: 0, transmission: 0.6, thickness: r,
            emissive: 0x0891b2, emissiveIntensity: 0.4, side: THREE.DoubleSide, ior: 1.35,
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(ip.x, ip.y - depth*0.4, ip.z);
        mesh.userData = {
            type: 'diffusion', taskId, injectionPointId: ip.id,
            radius_mm: r_mm, depth_mm, flow_rate: diff.flow_rate_mls, elapsed: diff.elapsed_seconds, data: diff,
        };
        const wireGeo = new THREE.IcosahedronGeometry(r * scale * 1.02, 2);
        const wireMat = new THREE.MeshBasicMaterial({ color: 0x22d3ee, wireframe: true, transparent: true, opacity: 0.25 });
        mesh.add(new THREE.Mesh(wireGeo, wireMat));
        this.scene.add(mesh);
        this.diffusionMeshes.push(mesh);
    }

    addStreamlines(pathlines) {
        if (!this.settings.showStreamlines) return;
        this.flowLineMgr.addStreamlines(pathlines);
    }

    addAllGroutingData(caveData, caveGroup) {
        let allPathlines = [];
        caveData.walls.forEach(wall => {
            (wall.grouting_tasks || []).forEach(task => {
                (task.latest_diffusions || []).forEach(diff => {
                    (task.injection_points || []).forEach(ip => {
                        if (ip.id === diff.injection_point_id) {
                            this.addDiffusionSphere(task.task_id, ip, diff);
                            if (diff.particle_pathlines) {
                                allPathlines = allPathlines.concat(diff.particle_pathlines);
                            }
                        }
                    });
                });
            });
        });
        if (allPathlines.length > 0) {
            this.addStreamlines(allPathlines);
        }
    }

    updateSettings() {
        this.diffusionMeshes.forEach(m => m.visible = this.settings.showDiffusion);
        this.flowLineMgr.setVisible(this.settings.showStreamlines && this.settings.showDiffusion);
    }

    update(time) {
        this.flowLineMgr.update(time);
    }

    dispose() {
        this.clear();
        this.flowLineMgr.dispose();
    }
}
