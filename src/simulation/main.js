import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { DragStateManager } from './utils/DragStateManager.js';
import { downloadExampleScenesFolder, getPosition, getQuaternion, toMujocoPos, reloadScene, reloadPolicy } from './mujocoUtils.js';
import { createDcMotors } from './dcMotor.js';

const defaultPolicy = "./examples/checkpoints/asimov/asimov_velocity_policy.json";

export class MuJoCoDemo {
  constructor(mujoco) {
    this.mujoco = mujoco;
    mujoco.FS.mkdir('/working');
    mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');

    this.params = {
      paused: true,
      current_motion: 'default',
      compliance_enabled: false,
      compliance_threshold: 10.0
    };
    this.policyRunner = null;
    this.kpPolicy = null;
    this.kdPolicy = null;
    this.actionTarget = null;
    this.dcMotors = null;
    this.velocityCommand = new Float32Array(3); // [vx, vy, wz]
    this.model = null;
    this.data = null;
    this.simulation = null;
    this.currentPolicyPath = defaultPolicy;

    this.bodies = {};
    this.lights = {};

    this.container = document.getElementById('mujoco-container');

    this.scene = new THREE.Scene();
    this.scene.name = 'scene';

    this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.001, 100);
    this.camera.name = 'PerspectiveCamera';
    this.camera.position.set(3.0, 2.2, 3.0);
    this.scene.add(this.camera);

    this.scene.background = new THREE.Color(0.15, 0.25, 0.35);
    this.scene.fog = null;

    this.ambientLight = new THREE.AmbientLight(0xffffff, 0.1);
    this.ambientLight.name = 'AmbientLight';
    this.scene.add(this.ambientLight);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderScale = 2.0;
    this.renderer.setPixelRatio(this.renderScale);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.simStepHz = 0;
    this._stepFrameCount = 0;
    this._stepLastTime = performance.now();
    this._lastRenderTime = 0;

    this.metrics = {
      pelvisZ: 0, gravZ: 0,
      bodyVx: 0, bodyVy: 0, angVelZ: 0,
      cmdVx: 0, cmdVy: 0, cmdWz: 0,
      speed: 0, velError: 0,
      heading: 0,
      jointRms: 0,
      footL: false, footR: false,
      totalPower: 0, totalTorque: 0,
      peakTorque: 0, peakJoint: '',
      cot: 0, mass: 0, fell: false,
      dragForce: 0, dragForceRaw: 0
    };
    this._totalMass = 0;
    this._footLGeoms = null;
    this._footRGeoms = null;

    // Sim2real debug state
    this._prevActions = null;       // for action rate
    this._reqTorques = null;        // PD-requested torques (before DC motor)
    this._appliedTorques = null;    // actual applied torques (after DC motor)
    this._footLHistory = [];        // for stance timing
    this._footRHistory = [];
    this._footHistoryLen = 100;     // ~2s at 50Hz

    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.7, 0);
    this.controls.panSpeed = 2;
    this.controls.zoomSpeed = 1;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.10;
    this.controls.screenSpacePanning = true;
    this.controls.update();

    window.addEventListener('resize', this.onWindowResize.bind(this));

    this.dragStateManager = new DragStateManager(this.scene, this.renderer, this.camera, this.container.parentElement, this.controls);

    this.followEnabled = false;
    this.followHeight = 0.75;
    this.followLerp = 0.05;
    this.followTarget = new THREE.Vector3();
    this.followTargetDesired = new THREE.Vector3();
    this.followDelta = new THREE.Vector3();
    this.followOffset = new THREE.Vector3();
    this.followInitialized = false;
    this.followBodyId = null;
    this.followDistance = this.camera.position.distanceTo(this.controls.target);

    this.lastSimState = {
      bodies: new Map(),
      lights: new Map(),
      tendons: {
        numWraps: 0,
        matrix: new THREE.Matrix4()
      }
    };

    this.renderer.setAnimationLoop(this.render.bind(this));

    this.reloadScene = reloadScene.bind(this);
    this.reloadPolicy = reloadPolicy.bind(this);
  }

  async init() {
    await downloadExampleScenesFolder(this.mujoco);
    // Load the scene specified by the default policy config
    const configResp = await fetch(defaultPolicy);
    const config = await configResp.json();
    const scene = config.scene ?? 'asimov/asimov_full.xml';
    await this.reloadScene(scene);
    this.updateFollowBodyId();
    await this.reloadPolicy(defaultPolicy);
    this.alive = true;
  }

  setVelocityCommand(vx, vy, wz) {
    this.velocityCommand[0] = vx;
    this.velocityCommand[1] = vy;
    this.velocityCommand[2] = wz;
    if (this.policyRunner) {
      this.policyRunner.velocityCommand = this.velocityCommand;
    }
  }

  async reload(mjcf_path) {
    await this.reloadScene(mjcf_path);
    this.updateFollowBodyId();
    this.timestep = this.model.opt.timestep;
    this.decimation = Math.max(1, Math.round(0.02 / this.timestep));

    console.log('timestep:', this.timestep, 'decimation:', this.decimation);

    await this.reloadPolicy(this.currentPolicyPath ?? defaultPolicy);
    this.alive = true;
  }

  setFollowEnabled(enabled) {
    this.followEnabled = Boolean(enabled);
    this.followInitialized = false;
    if (this.followEnabled) {
      this.followOffset.subVectors(this.camera.position, this.controls.target);
      if (this.followOffset.lengthSq() === 0) {
        this.followOffset.set(0, 0, 1);
      }
      this.followOffset.setLength(this.followDistance);
      this.camera.position.copy(this.controls.target).add(this.followOffset);
      this.controls.update();
    }
  }

  updateFollowBodyId() {
    if (Number.isInteger(this.pelvis_body_id)) {
      this.followBodyId = this.pelvis_body_id;
      return;
    }
    if (this.model && this.model.nbody > 1) {
      this.followBodyId = 1;
    }
  }

  updateCameraFollow() {
    if (!this.followEnabled) {
      return;
    }
    const bodyId = Number.isInteger(this.followBodyId) ? this.followBodyId : null;
    if (bodyId === null) {
      return;
    }
    const cached = this.lastSimState.bodies.get(bodyId);
    if (!cached) {
      return;
    }
    this.followTargetDesired.set(cached.position.x, this.followHeight, cached.position.z);
    if (!this.followInitialized) {
      this.followTarget.copy(this.followTargetDesired);
      this.followInitialized = true;
    } else {
      this.followTarget.lerp(this.followTargetDesired, this.followLerp);
    }

    this.followDelta.subVectors(this.followTarget, this.controls.target);
    this.controls.target.copy(this.followTarget);
    this.camera.position.add(this.followDelta);
  }

  async main_loop() {
    if (!this.policyRunner) {
      return;
    }

    while (this.alive) {
      const loopStart = performance.now();

      if (!this.params.paused && this.model && this.data && this.simulation && this.policyRunner) {
        const state = this.readPolicyState();

        try {
          this.actionTarget = await this.policyRunner.step(state);
        } catch (e) {
          console.error('Inference error in main loop:', e);
          this.alive = false;
          break;
        }

        // Allocate torque tracking arrays once
        if (!this._reqTorques || this._reqTorques.length !== this.numActions) {
          this._reqTorques = new Float32Array(this.numActions);
          this._appliedTorques = new Float32Array(this.numActions);
        }

        for (let substep = 0; substep < this.decimation; substep++) {
          if (this.control_type === 'joint_position') {
            for (let i = 0; i < this.numActions; i++) {
              const qpos_adr = this.qpos_adr_policy[i];
              const qvel_adr = this.qvel_adr_policy[i];
              const ctrl_adr = this.ctrl_adr_policy[i];

              const targetJpos = this.actionTarget ? this.actionTarget[i] : 0.0;
              const kp = this.kpPolicy ? this.kpPolicy[i] : 0.0;
              const kd = this.kdPolicy ? this.kdPolicy[i] : 0.0;
              const currentVel = this.simulation.qvel[qvel_adr];
              const torque = kp * (targetJpos - this.simulation.qpos[qpos_adr]) + kd * (0 - currentVel);

              this._reqTorques[i] = torque;

              let ctrlValue;
              if (this.dcMotors && this.dcMotors[i]) {
                ctrlValue = this.dcMotors[i].apply(torque, currentVel);
              } else {
                ctrlValue = torque;
                const ctrlRange = this.model?.actuator_ctrlrange;
                if (ctrlRange && ctrlRange.length >= (ctrl_adr + 1) * 2) {
                  const min = ctrlRange[ctrl_adr * 2];
                  const max = ctrlRange[(ctrl_adr * 2) + 1];
                  if (Number.isFinite(min) && Number.isFinite(max) && min < max) {
                    ctrlValue = Math.min(Math.max(ctrlValue, min), max);
                  }
                }
              }
              this._appliedTorques[i] = ctrlValue;
              this.simulation.ctrl[ctrl_adr] = ctrlValue;
            }
          } else if (this.control_type === 'torque') {
            console.error('Torque control not implemented yet.');
          }

          const applied = this.simulation.qfrc_applied;
          for (let i = 0; i < applied.length; i++) {
            applied[i] = 0.0;
          }

          const dragged = this.dragStateManager.physicsObject;
          if (dragged && dragged.bodyID) {
            for (let b = 0; b < this.model.nbody; b++) {
              if (this.bodies[b]) {
                getPosition(this.simulation.xpos, b, this.bodies[b].position);
                getQuaternion(this.simulation.xquat, b, this.bodies[b].quaternion);
                this.bodies[b].updateWorldMatrix();
              }
            }
            const bodyID = dragged.bodyID;
            this.dragStateManager.update();
            const force = toMujocoPos(
              this.dragStateManager.currentWorld.clone()
                .sub(this.dragStateManager.worldHit)
                .multiplyScalar(60.0)
            );
            // clamp force magnitude
            const forceMagnitude = Math.sqrt(force.x * force.x + force.y * force.y + force.z * force.z);
            const maxForce = 30.0;
            if (forceMagnitude > maxForce) {
              const scale = maxForce / forceMagnitude;
              force.x *= scale;
              force.y *= scale;
              force.z *= scale;
            }
            const point = toMujocoPos(this.dragStateManager.worldHit.clone());
            this.simulation.applyForce(force.x, force.y, force.z, 0, 0, 0, point.x, point.y, point.z, bodyID);
          }

          this.simulation.step();
        }

        for (let b = 0; b < this.model.nbody; b++) {
          if (!this.bodies[b]) {
            continue;
          }
          if (!this.lastSimState.bodies.has(b)) {
            this.lastSimState.bodies.set(b, {
              position: new THREE.Vector3(),
              quaternion: new THREE.Quaternion()
            });
          }
          const cached = this.lastSimState.bodies.get(b);
          getPosition(this.simulation.xpos, b, cached.position);
          getQuaternion(this.simulation.xquat, b, cached.quaternion);
        }

        const numLights = this.model.nlight;
        for (let l = 0; l < numLights; l++) {
          if (!this.lights[l]) {
            continue;
          }
          if (!this.lastSimState.lights.has(l)) {
            this.lastSimState.lights.set(l, {
              position: new THREE.Vector3(),
              direction: new THREE.Vector3()
            });
          }
          const cached = this.lastSimState.lights.get(l);
          getPosition(this.simulation.light_xpos, l, cached.position);
          getPosition(this.simulation.light_xdir, l, cached.direction);
        }

        this.lastSimState.tendons.numWraps = {
          count: this.model.nwrap,
          matrix: this.lastSimState.tendons.matrix
        };

        this.computeMetrics();

        this._stepFrameCount += 1;
        const now = performance.now();
        const elapsedStep = now - this._stepLastTime;
        if (elapsedStep >= 500) {
          this.simStepHz = (this._stepFrameCount * 1000) / elapsedStep;
          this._stepFrameCount = 0;
          this._stepLastTime = now;
        }
      } else {
        this.simStepHz = 0;
        this._stepFrameCount = 0;
        this._stepLastTime = performance.now();
      }

      const loopEnd = performance.now();
      const elapsed = (loopEnd - loopStart) / 1000;
      const target = this.timestep * this.decimation;
      const sleepTime = Math.max(0, target - elapsed);
      await new Promise((resolve) => setTimeout(resolve, sleepTime * 1000));
    }
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setPixelRatio(this.renderScale);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this._lastRenderTime = 0;
    this.render();
  }

  setRenderScale(scale) {
    const clamped = Math.max(0.5, Math.min(2.0, scale));
    this.renderScale = clamped;
    this.renderer.setPixelRatio(this.renderScale);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this._lastRenderTime = 0;
    this.render();
  }

  getSimStepHz() {
    return this.simStepHz;
  }

  readPolicyState() {
    const qpos = this.simulation.qpos;
    const qvel = this.simulation.qvel;
    const jointPos = new Float32Array(this.numActions);
    const jointVel = new Float32Array(this.numActions);
    for (let i = 0; i < this.numActions; i++) {
      const qposAdr = this.qpos_adr_policy[i];
      const qvelAdr = this.qvel_adr_policy[i];
      jointPos[i] = qpos[qposAdr];
      jointVel[i] = qvel[qvelAdr];
    }
    const rootPos = new Float32Array([qpos[0], qpos[1], qpos[2]]);
    const rootQuat = new Float32Array([qpos[3], qpos[4], qpos[5], qpos[6]]);
    const rootAngVel = new Float32Array([qvel[3], qvel[4], qvel[5]]);
    const complianceEnabled = Boolean(this.params?.compliance_enabled);
    const rawThreshold = Number(this.params?.compliance_threshold);
    const complianceThreshold = Number.isFinite(rawThreshold) ? rawThreshold : 10.0;
    return {
      jointPos,
      jointVel,
      rootPos,
      rootQuat,
      rootAngVel,
      complianceEnabled,
      complianceThreshold
    };
  }

  computeMetrics() {
    if (!this.simulation || !this.model) return;
    const qpos = this.simulation.qpos;
    const qvel = this.simulation.qvel;
    const ctrl = this.simulation.ctrl;
    const n = this.numActions || 0;
    const jointNames = this.policyRunner?.config?.policy_joint_names;

    // Pelvis height
    const pelvisZ = qpos[2];

    // Projected gravity from root quaternion (w,x,y,z at qpos[3..6])
    const qw = qpos[3], qx = qpos[4], qy = qpos[5], qz = qpos[6];
    const gravZ = -(1 - 2 * (qx * qx + qy * qy));

    // Body-frame velocity: rotate world lin vel by inverse root quat
    const vx_w = qvel[0], vy_w = qvel[1], vz_w = qvel[2];
    const bodyVx = vx_w * (1 - 2*(qy*qy + qz*qz)) + vy_w * 2*(qx*qy + qw*qz) + vz_w * 2*(qx*qz - qw*qy);
    const bodyVy = vx_w * 2*(qx*qy - qw*qz) + vy_w * (1 - 2*(qx*qx + qz*qz)) + vz_w * 2*(qy*qz + qw*qx);
    const angVelZ = qvel[5];

    // Commanded velocity
    const cmdVx = this.velocityCommand[0];
    const cmdVy = this.velocityCommand[1];
    const cmdWz = this.velocityCommand[2];

    // Speed & velocity tracking error
    const speed = Math.sqrt(bodyVx * bodyVx + bodyVy * bodyVy);
    const velError = Math.sqrt((cmdVx - bodyVx) ** 2 + (cmdVy - bodyVy) ** 2);

    // Heading (yaw from quaternion)
    const heading = Math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));

    // Joint tracking RMS (target vs actual)
    let jointSumSq = 0;
    if (this.actionTarget && n > 0) {
      for (let i = 0; i < n; i++) {
        const err = this.actionTarget[i] - qpos[this.qpos_adr_policy[i]];
        jointSumSq += err * err;
      }
    }
    const jointRms = n > 0 ? Math.sqrt(jointSumSq / n) : 0;

    // Torque / power from actuator ctrl and joint vel
    let totalPower = 0;
    let totalTorque = 0;
    let peakTorque = 0;
    let peakIdx = 0;
    for (let i = 0; i < n; i++) {
      const ctrlAdr = this.ctrl_adr_policy[i];
      const qvelAdr = this.qvel_adr_policy[i];
      const tau = Math.abs(ctrl[ctrlAdr]);
      const vel = Math.abs(qvel[qvelAdr]);
      totalTorque += tau;
      totalPower += tau * vel;
      if (tau > peakTorque) {
        peakTorque = tau;
        peakIdx = i;
      }
    }

    // Total mass (cache once)
    if (this._totalMass === 0 && this.model.nbody > 0) {
      let m = 0;
      for (let b = 0; b < this.model.nbody; b++) {
        m += this.model.body_mass[b];
      }
      this._totalMass = m;
    }
    const mass = this._totalMass;

    // Cost of transport
    const cot = speed > 0.05 ? totalPower / (mass * 9.81 * speed) : 0;

    // Foot contact via body z-height (approximate — check ankle_roll_link z)
    let footL = false;
    let footR = false;
    if (this._footLBodyId == null) {
      // Cache foot body IDs on first call
      this._footLBodyId = -1;
      this._footRBodyId = -1;
      if (this.model.nbody > 0) {
        for (let b = 0; b < this.model.nbody; b++) {
          const cached = this.lastSimState.bodies.get(b);
          if (!cached) continue;
          const name = this.bodies[b]?.name;
          if (name === 'left_ankle_roll_link') this._footLBodyId = b;
          if (name === 'right_ankle_roll_link') this._footRBodyId = b;
        }
      }
    }
    const footThreshold = 0.045;
    if (this._footLBodyId >= 0) {
      const c = this.lastSimState.bodies.get(this._footLBodyId);
      if (c) footL = c.position.y < footThreshold; // mujoco z → three.js y
    }
    if (this._footRBodyId >= 0) {
      const c = this.lastSimState.bodies.get(this._footRBodyId);
      if (c) footR = c.position.y < footThreshold;
    }

    // Fall detection
    const fell = pelvisZ < 0.3 || gravZ > -0.5;

    // --- Sim2Real Debug Metrics ---

    // Motor saturation: count joints where DC motor clipped significantly
    let satCount = 0;
    let satWorstJoint = '';
    let satWorstRatio = 0;
    if (this._reqTorques && this._appliedTorques && n > 0) {
      for (let i = 0; i < n; i++) {
        const req = Math.abs(this._reqTorques[i]);
        const app = Math.abs(this._appliedTorques[i]);
        if (req > 1.0 && app < req * 0.9) {
          satCount++;
          const ratio = 1 - app / req;
          if (ratio > satWorstRatio) {
            satWorstRatio = ratio;
            const jn = jointNames?.[i] ?? '';
            satWorstJoint = jn.replace('_joint', '').replace('left_', 'L_').replace('right_', 'R_');
          }
        }
      }
    }

    // Action rate (smoothness): L2 norm of action change per step
    let actionRate = 0;
    if (this.actionTarget && n > 0) {
      if (!this._prevActions) {
        this._prevActions = new Float32Array(n);
        if (this.actionTarget) this._prevActions.set(this.actionTarget);
      } else {
        let sumSq = 0;
        for (let i = 0; i < n; i++) {
          const d = this.actionTarget[i] - this._prevActions[i];
          sumSq += d * d;
        }
        actionRate = Math.sqrt(sumSq);
        this._prevActions.set(this.actionTarget);
      }
    }

    // Joint limit proximity: worst joint as % of range used
    let limitWorstPct = 0;
    let limitWorstJoint = '';
    if (n > 0 && this.model.jnt_range) {
      for (let i = 0; i < n; i++) {
        const qposAdr = this.qpos_adr_policy[i];
        const pos = qpos[qposAdr];
        // Find the joint index for this qpos address to get range
        const jntIdx = this.policyJointNames ? this._policyJointIndices?.[i] : -1;
        if (jntIdx >= 0) {
          const lo = this.model.jnt_range[jntIdx * 2];
          const hi = this.model.jnt_range[jntIdx * 2 + 1];
          if (hi > lo) {
            const range = hi - lo;
            const distToLimit = Math.min(pos - lo, hi - pos);
            const pct = 1 - distToLimit / (range / 2);
            if (pct > limitWorstPct) {
              limitWorstPct = pct;
              const jn = jointNames?.[i] ?? '';
              limitWorstJoint = jn.replace('_joint', '').replace('left_', 'L_').replace('right_', 'R_');
            }
          }
        }
      }
    }

    // Cache joint indices for limit check (once)
    if (!this._policyJointIndices && this.policyJointNames && this.model) {
      this._policyJointIndices = [];
      for (const name of this.policyJointNames) {
        const idx = this.jointNamesMJC?.indexOf(name) ?? -1;
        this._policyJointIndices.push(idx);
      }
    }

    // L-R torque asymmetry (legs only: indices 0-5 left, 6-11 right)
    let lrAsym = 0;
    if (this._appliedTorques && n >= 12) {
      let lSum = 0, rSum = 0;
      for (let i = 0; i < 6; i++) {
        lSum += Math.abs(this._appliedTorques[i]);
        rSum += Math.abs(this._appliedTorques[i + 6]);
      }
      const avg = (lSum + rSum) / 2;
      lrAsym = avg > 1.0 ? Math.abs(lSum - rSum) / avg : 0;
    }

    // Foot stance timing from contact history
    this._footLHistory.push(footL ? 1 : 0);
    this._footRHistory.push(footR ? 1 : 0);
    if (this._footLHistory.length > this._footHistoryLen) this._footLHistory.shift();
    if (this._footRHistory.length > this._footHistoryLen) this._footRHistory.shift();
    const stanceL = this._footLHistory.length > 0
      ? this._footLHistory.reduce((a, b) => a + b, 0) / this._footLHistory.length : 0;
    const stanceR = this._footRHistory.length > 0
      ? this._footRHistory.reduce((a, b) => a + b, 0) / this._footRHistory.length : 0;

    // Drag force magnitude
    let dragForceRaw = 0;
    const dragged = this.dragStateManager?.physicsObject;
    if (dragged && dragged.bodyID) {
      const dm = this.dragStateManager;
      if (dm.currentWorld && dm.worldHit) {
        const dx = dm.currentWorld.x - dm.worldHit.x;
        const dy = dm.currentWorld.y - dm.worldHit.y;
        const dz = dm.currentWorld.z - dm.worldHit.z;
        dragForceRaw = Math.min(Math.sqrt(dx*dx + dy*dy + dz*dz) * 60.0, 30.0);
      }
    }

    // Joint name for peak torque
    let peakJoint = '';
    if (jointNames && jointNames[peakIdx]) {
      peakJoint = jointNames[peakIdx]
        .replace('_joint', '')
        .replace('left_', 'L_')
        .replace('right_', 'R_');
    }

    this.metrics = {
      pelvisZ: pelvisZ.toFixed(3),
      gravZ: gravZ.toFixed(3),
      bodyVx: bodyVx.toFixed(2),
      bodyVy: bodyVy.toFixed(2),
      angVelZ: angVelZ.toFixed(2),
      cmdVx: cmdVx.toFixed(2),
      cmdVy: cmdVy.toFixed(2),
      cmdWz: cmdWz.toFixed(2),
      speed: speed.toFixed(2),
      velError: velError.toFixed(3),
      heading: (heading * 180 / Math.PI).toFixed(1),
      jointRms: (jointRms * 180 / Math.PI).toFixed(2),
      footL,
      footR,
      stanceL: (stanceL * 100).toFixed(0),
      stanceR: (stanceR * 100).toFixed(0),
      totalPower: totalPower.toFixed(1),
      totalTorque: totalTorque.toFixed(1),
      peakTorque: peakTorque.toFixed(1),
      peakJoint,
      cot: cot.toFixed(2),
      mass: mass.toFixed(1),
      satCount,
      satPct: n > 0 ? ((satCount / n) * 100).toFixed(0) : '0',
      satWorstJoint,
      actionRate: actionRate.toFixed(4),
      limitPct: (limitWorstPct * 100).toFixed(0),
      limitJoint: limitWorstJoint,
      lrAsym: (lrAsym * 100).toFixed(0),
      fell,
      dragForce: dragForceRaw.toFixed(1),
      dragForceRaw
    };
  }

  resetSimulation() {
    if (!this.simulation) {
      return;
    }
    this.params.paused = true;
    this.simulation.resetData();
    this.simulation.forward();
    this.actionTarget = null;
    if (this.policyRunner) {
      const state = this.readPolicyState();
      this.policyRunner.reset(state);
      this.params.current_motion = 'default';
    }
    this.params.paused = false;
  }

  render() {
    if (!this.model || !this.data || !this.simulation) {
      return;
    }
    const now = performance.now();
    if (now - this._lastRenderTime < 30) {
      return;
    }
    this._lastRenderTime = now;

    this.updateCameraFollow();
    this.controls.update();

    for (const [b, cached] of this.lastSimState.bodies) {
      if (this.bodies[b]) {
        this.bodies[b].position.copy(cached.position);
        this.bodies[b].quaternion.copy(cached.quaternion);
        this.bodies[b].updateWorldMatrix();
      }
    }

    for (const [l, cached] of this.lastSimState.lights) {
      if (this.lights[l]) {
        this.lights[l].position.copy(cached.position);
        this.lights[l].lookAt(cached.direction.clone().add(this.lights[l].position));
      }
    }

    if (this.mujocoRoot && this.mujocoRoot.cylinders) {
      const numWraps = this.lastSimState.tendons.numWraps.count;
      this.mujocoRoot.cylinders.count = numWraps;
      this.mujocoRoot.spheres.count = numWraps > 0 ? numWraps + 1 : 0;
      this.mujocoRoot.cylinders.instanceMatrix.needsUpdate = true;
      this.mujocoRoot.spheres.instanceMatrix.needsUpdate = true;
    }

    this.renderer.render(this.scene, this.camera);
  }
}
