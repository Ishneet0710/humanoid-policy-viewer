import * as ort from 'onnxruntime-web';
import { ONNXModule } from './onnxHelper.js';
import { Observations } from './observationHelpers.js';
import { toFloatArray } from './utils/math.js';

export class PolicyRunner {
  constructor(config, options = {}) {
    this.config = config;
    this.policyJointNames = (options.policyJointNames ?? config.policy_joint_names ?? []).slice();
    if (this.policyJointNames.length === 0) {
      throw new Error('PolicyRunner requires policy_joint_names in config');
    }
    this.numActions = this.policyJointNames.length;

    this.actionScale = toFloatArray(options.actionScale ?? config.action_scale, this.numActions, 1.0);
    this.defaultJointPos = toFloatArray(options.defaultJointPos ?? [], this.numActions, 0.0);
    this.actionClip = typeof config.action_clip === 'number' ? config.action_clip : 10.0;

    this.module = new ONNXModule(config.onnx);
    this.inputDict = {};
    this.isInferencing = false;
    this.lastActions = new Float32Array(this.numActions);
    this.velocityCommand = new Float32Array(3); // [vx, vy, wz] for velocity policies

    // Low-pass filter on actions (matches deploy_asimov.py --lpf_cutoff 10)
    const lpfCfg = config.lpf_cutoff ?? (config.robot_type === 'asimov_velocity' ? 10.0 : 0);
    this.lpfCutoff = lpfCfg > 0 ? lpfCfg : 0;
    this.lpfCtrlDt = 0.02; // SIMULATION_DT * CONTROL_DECIMATION = 0.005 * 4
    this.lpfYLast = new Float32Array(this.numActions);

    this.tracking = null;

    this.obsModules = this._buildObsModules(config.obs_config);
    this.numObs = this.obsModules.reduce((sum, obs) => sum + (obs.size ?? 0), 0);
  }

  async init() {
    await this.module.init();
    this.reset();
  }

  _buildObsModules(obsConfig) {
    const obsList = (obsConfig && Array.isArray(obsConfig.policy)) ? obsConfig.policy : [];
    return obsList.map((obsConfigEntry) => {
      const ObsClass = Observations[obsConfigEntry.name];
      if (!ObsClass) {
        throw new Error(`Unknown observation type: ${obsConfigEntry.name}`);
      }
      const kwargs = { ...obsConfigEntry };
      delete kwargs.name;
      return new ObsClass(this, kwargs);
    });
  }

  reset(state = null) {
    this.inputDict = this.module.initInput() ?? {};
    this.lastActions.fill(0.0);
    this.lpfYLast.fill(0.0);
    if (this.tracking) {
      this.tracking.reset(state);
    }
    for (const obs of this.obsModules) {
      if (typeof obs.reset === 'function') {
        obs.reset(state);
      }
    }
  }

  async step(state) {
    if (this.isInferencing) {
      return null;
    }

    if (!state) {
      throw new Error('PolicyRunner.step requires a state object');
    }

    this.isInferencing = true;
    try {
      if (this.tracking) {
        this.tracking.advance();
      }

      const obsForPolicy = new Float32Array(this.numObs);
      let offset = 0;
      for (const obs of this.obsModules) {
        if (typeof obs.update === 'function') {
          obs.update(state);
        }
        const obsValue = obs.compute(state);
        const obsArray = ArrayBuffer.isView(obsValue) ? obsValue : Float32Array.from(obsValue);
        obsForPolicy.set(obsArray, offset);
        offset += obsArray.length;
      }

      const inputKey = this.config.onnx?.meta?.in_keys?.[0] ?? 'policy';
      this.inputDict[inputKey] = new ort.Tensor('float32', obsForPolicy, [1, obsForPolicy.length]);

      const [result, carry] = await this.module.runInference(this.inputDict);
      this.inputDict = { ...this.inputDict, ...carry };

      const actionKey = (this.config.onnx?.meta?.out_keys ?? []).find(
        (k) => k === 'action' || k === 'actions'
      ) ?? 'action';
      const action = result[actionKey]?.data;
      if (!action || action.length !== this.numActions) {
        throw new Error('PolicyRunner received invalid action output');
      }

      const clip = typeof this.actionClip === 'number' ? this.actionClip : Infinity;
      for (let i = 0; i < this.numActions; i++) {
        const value = action[i];
        this.lastActions[i] = clip !== Infinity ? Math.max(-clip, Math.min(clip, value)) : value;
      }

      // One-pole low-pass filter: y = y_last + alpha * (x - y_last)
      if (this.lpfCutoff > 0) {
        const rc = 1.0 / (2.0 * Math.PI * this.lpfCutoff);
        const alpha = this.lpfCtrlDt / (rc + this.lpfCtrlDt);
        for (let i = 0; i < this.numActions; i++) {
          this.lastActions[i] = this.lpfYLast[i] + alpha * (this.lastActions[i] - this.lpfYLast[i]);
          this.lpfYLast[i] = this.lastActions[i];
        }
      }

      const target = new Float32Array(this.numActions);
      for (let i = 0; i < this.numActions; i++) {
        target[i] = this.defaultJointPos[i] + this.actionScale[i] * this.lastActions[i];
      }

      return target;
    } finally {
      this.isInferencing = false;
    }
  }
}
