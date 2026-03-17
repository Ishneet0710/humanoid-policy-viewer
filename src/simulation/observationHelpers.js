import * as THREE from 'three';
import {
  normalizeQuat,
  quatMultiply,
  quatInverse,
  quatApplyInv,
  quatToRot6d,
  clampFutureIndices
} from './utils/math.js';

class BootIndicator {
  get size() {
    return 1;
  }

  compute() {
    return new Float32Array([0.0]);
  }
}

class ComplianceFlagObs {
  get size() {
    return 3;
  }

  compute(state) {
    const enabled = state?.complianceEnabled ? 1.0 : 0.0;
    const rawThreshold = Number(state?.complianceThreshold);
    const threshold = Number.isFinite(rawThreshold) ? rawThreshold : 0.0;
    const kp = threshold / 0.05;
    return new Float32Array([enabled, enabled * threshold, enabled * kp]);
  }
}

class RootAngVelB {
  get size() {
    return 3;
  }

  compute(state) {
    return new Float32Array(state.rootAngVel);
  }
}

class ProjectedGravityB {
  constructor() {
    this.gravity = new THREE.Vector3(0, 0, -1);
  }

  get size() {
    return 3;
  }

  compute(state) {
    const quat = state.rootQuat;
    const quatObj = new THREE.Quaternion(quat[1], quat[2], quat[3], quat[0]);
    const gravityLocal = this.gravity.clone().applyQuaternion(quatObj.clone().invert());
    return new Float32Array([gravityLocal.x, gravityLocal.y, gravityLocal.z]);
  }
}

class JointPos {
  constructor(policy, kwargs = {}) {
    const { pos_steps = [0, 1, 2, 3, 4, 8] } = kwargs;
    this.posSteps = pos_steps.slice();
    this.numJoints = policy.numActions;

    this.maxStep = Math.max(...this.posSteps);
    this.history = Array.from({ length: this.maxStep + 1 }, () => new Float32Array(this.numJoints));
  }

  get size() {
    return this.posSteps.length * this.numJoints;
  }

  reset(state) {
    const source = state?.jointPos ?? new Float32Array(this.numJoints);
    this.history[0].set(source);
    for (let i = 1; i < this.history.length; i++) {
      this.history[i].set(this.history[0]);
    }
  }

  update(state) {
    for (let i = this.history.length - 1; i > 0; i--) {
      this.history[i].set(this.history[i - 1]);
    }
    this.history[0].set(state.jointPos);
  }

  compute() {
    const out = new Float32Array(this.posSteps.length * this.numJoints);
    let offset = 0;
    for (const step of this.posSteps) {
      const idx = Math.min(step, this.history.length - 1);
      out.set(this.history[idx], offset);
      offset += this.numJoints;
    }
    return out;
  }
}

class TrackingCommandObsRaw {
  constructor(policy, kwargs = {}) {
    this.policy = policy;
    this.futureSteps = kwargs.future_steps ?? [0, 2, 4, 8, 16];
    const nFut = this.futureSteps.length;
    this.outputLength = (nFut - 1) * 3 + nFut * 6;
  }

  get size() {
    return this.outputLength;
  }

  compute(state) {
    const tracking = this.policy.tracking;
    if (!tracking || !tracking.isReady()) {
      return new Float32Array(this.outputLength);
    }

    const baseIdx = tracking.refIdx;
    const refLen = tracking.refLen;
    const indices = clampFutureIndices(baseIdx, this.futureSteps, refLen);

    const basePos = tracking.refRootPos[indices[0]];
    const baseQuat = normalizeQuat(tracking.refRootQuat[indices[0]]);

    const posDiff = [];
    for (let i = 1; i < indices.length; i++) {
      const pos = tracking.refRootPos[indices[i]];
      const diff = [pos[0] - basePos[0], pos[1] - basePos[1], pos[2] - basePos[2]];
      const diffB = quatApplyInv(baseQuat, diff);
      posDiff.push(diffB[0], diffB[1], diffB[2]);
    }

    const qCur = normalizeQuat(state.rootQuat);
    const qCurInv = quatInverse(qCur);

    const rot6d = [];
    for (let i = 0; i < indices.length; i++) {
      const refQuat = normalizeQuat(tracking.refRootQuat[indices[i]]);
      const rel = quatMultiply(qCurInv, refQuat);
      const r6 = quatToRot6d(rel);
      rot6d.push(r6[0], r6[1], r6[2], r6[3], r6[4], r6[5]);
    }

    return Float32Array.from([...posDiff, ...rot6d]);
  }
}

class TargetRootZObs {
  constructor(policy, kwargs = {}) {
    this.policy = policy;
    this.futureSteps = kwargs.future_steps ?? [0, 2, 4, 8, 16];
  }

  get size() {
    return this.futureSteps.length;
  }

  compute() {
    const tracking = this.policy.tracking;
    if (!tracking || !tracking.isReady()) {
      return new Float32Array(this.size);
    }
    const indices = clampFutureIndices(tracking.refIdx, this.futureSteps, tracking.refLen);
    const out = new Float32Array(indices.length);
    for (let i = 0; i < indices.length; i++) {
      out[i] = tracking.refRootPos[indices[i]][2] + 0.035;
    }
    return out;
  }
}

class TargetJointPosObs {
  constructor(policy, kwargs = {}) {
    this.policy = policy;
    this.futureSteps = kwargs.future_steps ?? [0, 2, 4, 8, 16];
  }

  get size() {
    const nJoints = this.policy.tracking?.nJoints ?? 0;
    return this.futureSteps.length * nJoints * 2;
  }

  compute(state) {
    const tracking = this.policy.tracking;
    if (!tracking || !tracking.isReady()) {
      return new Float32Array(this.size);
    }
    const indices = clampFutureIndices(tracking.refIdx, this.futureSteps, tracking.refLen);
    const out = new Float32Array(indices.length * tracking.nJoints);
    const outDiff = new Float32Array(indices.length * tracking.nJoints);
    const current = state?.jointPos ?? new Float32Array(tracking.nJoints);
    let offset = 0;
    for (const idx of indices) {
      const target = tracking.refJointPos[idx];
      out.set(target, offset);
      for (let j = 0; j < tracking.nJoints; j++) {
        outDiff[offset + j] = target[j] - (current[j] ?? 0.0);
      }
      offset += tracking.nJoints;
    }
    const merged = new Float32Array(out.length + outDiff.length);
    merged.set(out, 0);
    merged.set(outDiff, out.length);
    return merged;
  }
}

class TargetProjectedGravityBObs {
  constructor(policy, kwargs = {}) {
    this.policy = policy;
    this.futureSteps = kwargs.future_steps ?? [0, 2, 4, 8, 16];
  }

  get size() {
    return this.futureSteps.length * 3;
  }

  compute() {
    const tracking = this.policy.tracking;
    if (!tracking || !tracking.isReady()) {
      return new Float32Array(this.size);
    }
    const indices = clampFutureIndices(tracking.refIdx, this.futureSteps, tracking.refLen);
    const out = new Float32Array(indices.length * 3);
    const g = [0.0, 0.0, -1.0];
    let offset = 0;
    for (const idx of indices) {
      const quat = normalizeQuat(tracking.refRootQuat[idx]);
      const gLocal = quatApplyInv(quat, g);
      out[offset++] = gLocal[0];
      out[offset++] = gLocal[1];
      out[offset++] = gLocal[2];
    }
    return out;
  }
}


class PrevActions {
  /**
   * 
   * @param {mujoco.Model} model 
   * @param {mujoco.Simulation} simulation 
   * @param {MuJoCoDemo} demo
   * @param {number} steps 
   */
  constructor(policy, kwargs = {}) {
    this.policy = policy;
    const { history_steps = 4 } = kwargs;
    this.steps = Math.max(1, Math.floor(history_steps));
    this.numActions = policy.numActions;
    this.actionBuffer = Array.from({ length: this.steps }, () => new Float32Array(this.numActions));
  }

  /**
   * 
   * @param {dict} extra_info
   * @returns {Float32Array}
   */
  compute() {
    const flattened = new Float32Array(this.steps * this.numActions);
    for (let i = 0; i < this.steps; i++) {
      for (let j = 0; j < this.numActions; j++) {
        flattened[i * this.numActions + j] = this.actionBuffer[i][j];
      }
    }
    return flattened;
  }

  reset() {
    for (const buffer of this.actionBuffer) {
      buffer.fill(0.0);
    }
  }

  update() {
    for (let i = this.actionBuffer.length - 1; i > 0; i--) {
      this.actionBuffer[i].set(this.actionBuffer[i - 1]);
    }
    const source = this.policy?.lastActions ?? new Float32Array(this.numActions);
    this.actionBuffer[0].set(source);
  }

  get size() {
    return this.steps * this.numActions;
  }
}


/**
 * Asimov velocity-command observation (78 dims).
 *
 * Layout matches deploy_asimov.py exactly:
 *   [base_ang_vel(3), projected_gravity(3), command(3),
 *    joint_pos_slot01(9), joint_pos_slot23(8), joint_pos_slot45(6),
 *    joint_vel_slot01(9), joint_vel_slot23(8), joint_vel_slot45(6),
 *    prev_actions(23)]
 */
class AsimovVelocityObs {
  constructor(policy) {
    this.policy = policy;
    this.numActions = policy.numActions; // 23
    this.gravity = new THREE.Vector3(0, 0, -1);

    // CAN slot reorder indices (into XML-order joint arrays)
    this.obsJointReorder = [
      0, 1, 6, 7, 12, 13, 14, 18, 19,  // slot 0-1 (9)
      2, 3, 8, 9, 15, 16, 20, 21,       // slot 2-3 (8)
      4, 5, 10, 11, 17, 22              // slot 4-5 (6)
    ];
    this.slotSizes = [9, 8, 6];

    this.angVelScale = 0.25;
    this.jointVelScale = 0.1;

    // Velocity command (set externally via policy.velocityCommand)
    this.command = new Float32Array(3); // [vx, vy, wz]
  }

  get size() {
    return 78; // 3+3+3 + 9+8+6 + 9+8+6 + 23
  }

  reset() {
    this.command.fill(0.0);
  }

  compute(state) {
    const out = new Float32Array(78);
    let offset = 0;

    // 1. Base angular velocity * scale (3)
    const angVel = state.rootAngVel;
    out[offset++] = angVel[0] * this.angVelScale;
    out[offset++] = angVel[1] * this.angVelScale;
    out[offset++] = angVel[2] * this.angVelScale;

    // 2. Projected gravity (3)
    const quat = state.rootQuat; // [w, x, y, z]
    const w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    out[offset++] = 2.0 * (w * y - x * z);
    out[offset++] = -2.0 * (w * x + y * z);
    out[offset++] = -1.0 + 2.0 * (x * x + y * y);

    // 3. Velocity command (3)
    const cmd = this.policy.velocityCommand ?? this.command;
    out[offset++] = cmd[0];
    out[offset++] = cmd[1];
    out[offset++] = cmd[2];

    // 4. Joint positions relative to default, CAN-slot reordered
    const defaultPos = this.policy.defaultJointPos;
    const jointPos = state.jointPos;
    const jointVel = state.jointVel;

    // Compute relative positions
    const relPos = new Float32Array(this.numActions);
    for (let i = 0; i < this.numActions; i++) {
      relPos[i] = jointPos[i] - (defaultPos[i] ?? 0.0);
    }

    // Reorder and write joint positions by slot
    for (const idx of this.obsJointReorder) {
      out[offset++] = relPos[idx];
    }

    // 5. Joint velocities, CAN-slot reordered, scaled
    for (const idx of this.obsJointReorder) {
      out[offset++] = jointVel[idx] * this.jointVelScale;
    }

    // 6. Previous actions (23)
    const prevActions = this.policy.lastActions;
    for (let i = 0; i < this.numActions; i++) {
      out[offset++] = prevActions[i];
    }

    return out;
  }
}


// Export a dictionary of all observation classes
export const Observations = {
  PrevActions,
  BootIndicator,
  ComplianceFlagObs,
  RootAngVelB,
  ProjectedGravityB,
  JointPos,
  TrackingCommandObsRaw,
  TargetRootZObs,
  TargetJointPosObs,
  TargetProjectedGravityBObs,
  AsimovVelocityObs
};
