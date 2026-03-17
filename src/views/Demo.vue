<template>
  <div id="mujoco-container"></div>
  <div class="global-alerts">
    <v-alert
      v-if="isSmallScreen"
      v-model="showSmallScreenAlert"
      type="warning"
      variant="flat"
      density="compact"
      closable
      class="small-screen-alert"
    >
      Screen too small. The control panel is unavailable on small screens. Please use a desktop device.
    </v-alert>
    <v-alert
      v-if="isSafari"
      v-model="showSafariAlert"
      type="warning"
      variant="flat"
      density="compact"
      closable
      class="safari-alert"
    >
      Safari has lower memory limits, which can cause WASM to crash.
    </v-alert>
  </div>
  <div v-if="!isSmallScreen" class="controls">
    <v-card class="controls-card">
      <v-card-title>Asimov Policy Viewer</v-card-title>
      <v-card-text class="py-0 controls-body">
        <div class="metrics-grid">
          <span class="metric-label">Sim Hz</span>
          <span class="metric-value">{{ simStepLabel }}</span>
          <span class="metric-label">Height</span>
          <span class="metric-value">{{ metrics.pelvisZ }} m</span>
          <span class="metric-label">Grav Z</span>
          <span class="metric-value">{{ metrics.gravZ }}</span>
          <span class="metric-label">Heading</span>
          <span class="metric-value">{{ metrics.heading }}°</span>
          <span class="metric-label">Cmd Vel</span>
          <span class="metric-value">{{ metrics.cmdVx }} / {{ metrics.cmdVy }} / {{ metrics.cmdWz }}</span>
          <span class="metric-label">Body Vel</span>
          <span class="metric-value">{{ metrics.bodyVx }} / {{ metrics.bodyVy }} / {{ metrics.angVelZ }}</span>
          <span class="metric-label">Speed</span>
          <span class="metric-value">{{ metrics.speed }} m/s</span>
          <span class="metric-label">Vel Error</span>
          <span class="metric-value" :style="{ color: parseFloat(metrics.velError) > 0.2 ? '#ff9800' : '#4caf50' }">{{ metrics.velError }} m/s</span>
          <span class="metric-label">Joint RMS</span>
          <span class="metric-value">{{ metrics.jointRms }}°</span>
          <span class="metric-label">Feet</span>
          <span class="metric-value">
            <span :style="{ color: metrics.footL ? '#4caf50' : '#666' }">L</span>
            {{ ' ' }}
            <span :style="{ color: metrics.footR ? '#4caf50' : '#666' }">R</span>
            <span class="metric-dim"> ({{ metrics.stanceL }}/{{ metrics.stanceR }}%)</span>
          </span>
          <span class="metric-label">Power</span>
          <span class="metric-value">{{ metrics.totalPower }} W</span>
          <span class="metric-label">Torque</span>
          <span class="metric-value">{{ metrics.totalTorque }} Nm</span>
          <span class="metric-label">Peak</span>
          <span class="metric-value">{{ metrics.peakTorque }} ({{ metrics.peakJoint }})</span>
          <span class="metric-label">CoT</span>
          <span class="metric-value">{{ metrics.cot }}</span>
          <span class="metric-label">Mass</span>
          <span class="metric-value">{{ metrics.mass }} kg</span>
          <span class="metric-label">Motor Sat</span>
          <span class="metric-value" :style="{ color: parseInt(metrics.satPct) > 0 ? '#ff9800' : undefined }">
            {{ metrics.satPct }}%<span v-if="metrics.satWorstJoint" class="metric-dim"> ({{ metrics.satWorstJoint }})</span>
          </span>
          <span class="metric-label">Action Δ</span>
          <span class="metric-value" :style="{ color: parseFloat(metrics.actionRate) > 0.5 ? '#ff9800' : undefined }">{{ metrics.actionRate }}</span>
          <span class="metric-label">Jnt Limit</span>
          <span class="metric-value" :style="{ color: parseInt(metrics.limitPct) > 90 ? '#ff5252' : parseInt(metrics.limitPct) > 70 ? '#ff9800' : undefined }">
            {{ metrics.limitPct }}%<span v-if="metrics.limitJoint" class="metric-dim"> ({{ metrics.limitJoint }})</span>
          </span>
          <span class="metric-label">L-R Asym</span>
          <span class="metric-value" :style="{ color: parseInt(metrics.lrAsym) > 30 ? '#ff9800' : undefined }">{{ metrics.lrAsym }}%</span>
          <span class="metric-label">Ext Force</span>
          <span class="metric-value" :style="{ color: metrics.dragForceRaw > 0 ? '#ff5252' : undefined }">{{ metrics.dragForce }} N</span>
        </div>
        <v-chip v-if="metrics.fell" color="error" size="small" class="mt-1">FELL</v-chip>
        <v-divider class="my-2"/>
        <span class="status-name">Movement</span>
          <div class="movement-buttons">
            <v-btn
              v-for="mode in movementModes"
              :key="mode.name"
              size="small"
              :variant="activeMovement === mode.name ? 'flat' : 'tonal'"
              :color="activeMovement === mode.name ? 'primary' : undefined"
              :disabled="state !== 1"
              @click="setMovement(mode)"
            >
              {{ mode.label }}
            </v-btn>
          </div>

      </v-card-text>
      <v-card-actions>
        <v-btn color="primary" block @click="reset">Reset</v-btn>
      </v-card-actions>
    </v-card>
  </div>
  <v-dialog :model-value="state === 0" persistent max-width="600px" scrollable>
    <v-card title="Loading Simulation Environment">
      <v-card-text>
        <v-progress-linear indeterminate color="primary"></v-progress-linear>
        Loading MuJoCo and ONNX policy, please wait
      </v-card-text>
    </v-card>
  </v-dialog>
  <v-dialog :model-value="state < 0" persistent max-width="600px" scrollable>
    <v-card title="Simulation Environment Loading Error">
      <v-card-text>
        <span v-if="state === -1">
          Unexpected runtime error, please refresh the page.<br />
          {{ extra_error_message }}
        </span>
        <span v-else-if="state === -2">
          Your browser does not support WebAssembly. Please use a recent version of Chrome, Edge, or Firefox.
        </span>
      </v-card-text>
    </v-card>
  </v-dialog>
</template>

<script>
import { MuJoCoDemo } from '@/simulation/main.js';
import loadMujoco from 'mujoco-js';

export default {
  name: 'DemoPage',
  data: () => ({
    state: 0, // 0: loading, 1: running, -1: JS error, -2: wasm unsupported
    extra_error_message: '',
    keydown_listener: null,
    cmdVx: 0.0,
    cmdVy: 0.0,
    cmdWz: 0.0,
    activeMovement: 'stand',
    movementModes: [
      { name: 'stand', label: 'Stand', vx: 0.0, vy: 0.0 },
      { name: 'forward', label: 'Forward', vx: 0.5, vy: 0.0 },
      { name: 'backward', label: 'Backward', vx: -0.5, vy: 0.0 },
      { name: 'left', label: 'Left', vx: 0.0, vy: 0.5 },
      { name: 'right', label: 'Right', vx: 0.0, vy: -0.5 },
    ],
    simStepHz: 0,
    isSmallScreen: false,
    showSmallScreenAlert: true,
    isSafari: false,
    showSafariAlert: true,
    resize_listener: null,
    metricsTimer: null,
    metrics: {
      pelvisZ: '0.000', gravZ: '0.000',
      bodyVx: '0.00', bodyVy: '0.00', angVelZ: '0.00',
      cmdVx: '0.00', cmdVy: '0.00', cmdWz: '0.00',
      speed: '0.00', velError: '0.000',
      heading: '0.0',
      jointRms: '0.00',
      footL: false, footR: false,
      stanceL: '0', stanceR: '0',
      totalPower: '0.0', totalTorque: '0.0',
      peakTorque: '0.0', peakJoint: '',
      cot: '0.00', mass: '0.0',
      satCount: 0, satPct: '0', satWorstJoint: '',
      actionRate: '0.0000',
      limitPct: '0', limitJoint: '',
      lrAsym: '0',
      fell: false,
      dragForce: '0.0', dragForceRaw: 0
    }
  }),
  computed: {
    simStepLabel() {
      if (!this.simStepHz || !Number.isFinite(this.simStepHz)) {
        return '—';
      }
      return `${this.simStepHz.toFixed(1)} Hz`;
    }
  },
  methods: {
    detectSafari() {
      const ua = navigator.userAgent;
      return /Safari\//.test(ua)
        && !/Chrome\//.test(ua)
        && !/Chromium\//.test(ua)
        && !/Edg\//.test(ua)
        && !/OPR\//.test(ua)
        && !/SamsungBrowser\//.test(ua)
        && !/CriOS\//.test(ua)
        && !/FxiOS\//.test(ua);
    },
    updateScreenState() {
      const isSmall = window.innerWidth < 500 || window.innerHeight < 700;
      if (!isSmall && this.isSmallScreen) {
        this.showSmallScreenAlert = true;
      }
      this.isSmallScreen = isSmall;
    },
    async init() {
      if (typeof WebAssembly !== 'object' || typeof WebAssembly.instantiate !== 'function') {
        this.state = -2;
        return;
      }

      try {
        const mujoco = await loadMujoco();
        this.demo = new MuJoCoDemo(mujoco);
        await this.demo.init();
        this.demo.main_loop();
        this.demo.params.paused = false;
        this.startMetricsPoll();
        this.state = 1;
      } catch (error) {
        this.state = -1;
        this.extra_error_message = error.toString();
        console.error(error);
      }
    },
    onVelocityChange() {
      if (!this.demo) {
        return;
      }
      this.demo.setVelocityCommand(this.cmdVx, this.cmdVy, this.cmdWz);
    },
    setMovement(mode) {
      this.activeMovement = mode.name;
      this.cmdVx = mode.vx;
      this.cmdVy = mode.vy;
      this.cmdWz = 0.0;
      this.onVelocityChange();
    },
    reset() {
      if (!this.demo) {
        return;
      }
      this.demo.resetSimulation();
    },
    startMetricsPoll() {
      this.stopMetricsPoll();
      this.metricsTimer = setInterval(() => {
        if (!this.demo) return;
        this.simStepHz = this.demo.getSimStepHz?.() ?? this.demo.simStepHz ?? 0;
        if (this.demo.metrics) {
          this.metrics = { ...this.demo.metrics };
        }
      }, 33);
    },
    stopMetricsPoll() {
      if (this.metricsTimer) {
        clearInterval(this.metricsTimer);
        this.metricsTimer = null;
      }
    }
  },
  mounted() {
    this.isSafari = this.detectSafari();
    this.updateScreenState();
    this.resize_listener = () => this.updateScreenState();
    window.addEventListener('resize', this.resize_listener);
    this.init();
    this.keydown_listener = (event) => {
      if (event.code === 'Backspace') this.reset();
    };
    document.addEventListener('keydown', this.keydown_listener);
  },
  beforeUnmount() {
    this.stopMetricsPoll();
    document.removeEventListener('keydown', this.keydown_listener);
    if (this.resize_listener) {
      window.removeEventListener('resize', this.resize_listener);
    }
  }
};
</script>

<style scoped>
.controls {
  position: fixed;
  top: 20px;
  right: 20px;
  width: 320px;
  z-index: 1000;
}

.global-alerts {
  position: fixed;
  top: 20px;
  left: 16px;
  right: 16px;
  max-width: 520px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
  z-index: 1200;
}

.small-screen-alert {
  width: 100%;
}

.safari-alert {
  width: 100%;
}

.controls-card {
  max-height: calc(100vh - 40px);
}

.controls-body {
  max-height: calc(100vh - 160px);
  overflow-y: auto;
  overscroll-behavior: contain;
}

.status-name {
  font-weight: 600;
}

.movement-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 4px;
}

.metrics-grid {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 2px 10px;
  font-size: 0.8rem;
  font-family: 'Roboto Mono', monospace;
  line-height: 1.5;
}

.metric-label {
  color: rgba(0, 0, 0, 0.5);
  font-weight: 500;
}

.metric-value {
  text-align: right;
  color: rgba(0, 0, 0, 0.87);
}

.metric-dim {
  color: rgba(0, 0, 0, 0.4);
  font-size: 0.75rem;
}
</style>
