/**
 * DC motor model with torque-speed curve + friction.
 * Matches deploy_asimov.py DcMotorModel exactly.
 */
export class DcMotorModel {
  constructor({ Y1, Y2, X1, X2, Fs, Fd, Va = 0.01, effort_limit = Infinity }) {
    this.Y1 = Y1;
    this.Y2 = Y2;
    this.X1 = X1;
    this.X2 = X2;
    this.Fs = Fs;
    this.Fd = Fd;
    this.Va = Va;
    this.effortLimit = effort_limit;
  }

  apply(torque, velocity) {
    const friction = this.Fs * Math.tanh(velocity / this.Va) + this.Fd * velocity;
    let effort = torque - friction;
    const sameDirection = (velocity * effort) > 0;
    const Y = sameDirection ? this.Y1 : this.Y2;
    const absVel = Math.abs(velocity);
    let limit;
    if (absVel < this.X1) {
      limit = Y;
    } else if (absVel < this.X2) {
      limit = Y * (this.X2 - absVel) / (this.X2 - this.X1);
    } else {
      limit = 0.0;
    }
    const clipped = Math.max(-limit, Math.min(limit, effort));
    return Math.max(-this.effortLimit, Math.min(this.effortLimit, clipped));
  }
}

/**
 * Create array of DcMotorModel instances from policy config.
 */
export function createDcMotors(dcConfig) {
  if (!dcConfig) {
    return null;
  }
  const motorTypes = dcConfig.joint_motor_types;
  const motorDefs = dcConfig.motors;
  if (!motorTypes || !motorDefs) {
    return null;
  }
  return motorTypes.map((type) => {
    const params = motorDefs[type];
    if (!params) {
      throw new Error(`Unknown motor type: ${type}`);
    }
    return new DcMotorModel(params);
  });
}
