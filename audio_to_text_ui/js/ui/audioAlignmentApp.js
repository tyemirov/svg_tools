// @ts-check

import {
  AUDIO_INPUT_ACCEPT,
  DEFAULT_LANGUAGE,
  DEFAULT_REMOVE_PUNCTUATION,
  LANGUAGE_OPTIONS,
  SSE_RECONNECT_DELAY_MS,
  STATUS_LABELS,
  TEXT_INPUT_ACCEPT,
  UI_STRINGS,
} from "../constants.js";
import { createBackendClient } from "../core/backendClient.js";
import { logError } from "../utils/logging.js";

/**
 * @param {File} file
 * @returns {Promise<string | null>}
 */
async function detectLanguageFromTextFile(file) {
  if (!file || typeof file.slice !== "function" || typeof FileReader === "undefined") {
    return null;
  }
  const maxBytes = 8192;
  const slice = file.slice(0, maxBytes);
  const reader = new FileReader();
  const buffer = await new Promise((resolve) => {
    reader.onerror = () => resolve(null);
    reader.onload = () => resolve(reader.result || null);
    reader.readAsArrayBuffer(slice);
  });
  if (!buffer || !(buffer instanceof ArrayBuffer)) {
    return null;
  }
  let text = "";
  try {
    const decoder = new TextDecoder("utf-8", { fatal: false });
    text = decoder.decode(buffer);
  } catch (error) {
    return null;
  }
  const hasCyrillic = /[\u0400-\u04FF]/.test(text);
  if (hasCyrillic) {
    return "ru";
  }
  return null;
}

/**
 * @returns {object}
 */
export function AudioAlignmentApp() {
  return {
    strings: UI_STRINGS,
    audioAccept: AUDIO_INPUT_ACCEPT,
    textAccept: TEXT_INPUT_ACCEPT,
    languageOptions: LANGUAGE_OPTIONS,
    language: DEFAULT_LANGUAGE,
    removePunctuation: DEFAULT_REMOVE_PUNCTUATION,
    audioFile: null,
    textFile: null,
    audioMeta: UI_STRINGS.audioHint,
    textMeta: UI_STRINGS.textHint,
    audioDragging: false,
    textDragging: false,
    statusLine: UI_STRINGS.statusReady,
    statusSub: UI_STRINGS.statusSub,
    errorMessage: "",
    jobs: [],
    jobSubmitting: false,
    jobStream: null,
    backend: null,
    pendingSubmission: null,
    languageTouched: false,
    init() {
      this.backend = createBackendClient();
      this.restoreLanguage();
      this.loadJobs();
      this.startJobStream();
    },
    get audioFilled() {
      return Boolean(this.audioFile);
    },
    get textFilled() {
      return Boolean(this.textFile);
    },
    get canRun() {
      return !this.jobSubmitting && Boolean(this.audioFile) && Boolean(this.textFile);
    },
    get jobCount() {
      return this.jobs.length;
    },
    setStatus(mainText, subText) {
      this.statusLine = mainText;
      this.statusSub = subText;
    },
    setError(message) {
      this.errorMessage = message;
    },
    clearError() {
      this.errorMessage = "";
    },
    setDragState(kind, value) {
      if (kind === "audio") {
        this.audioDragging = value;
        return;
      }
      this.textDragging = value;
    },
    handleDrop(kind, event) {
      const files = event.dataTransfer?.files;
      if (!files || files.length === 0) {
        this.setDragState(kind, false);
        return;
      }
      const file = files[0];
      if (kind === "audio") {
        this.assignAudioFile(file);
      } else {
        this.assignTextFile(file);
      }
      this.setDragState(kind, false);
    },
    handleAudioChange(event) {
      const file = event.target?.files && event.target.files[0];
      if (file) {
        this.assignAudioFile(file);
      }
    },
    handleTextChange(event) {
      const file = event.target?.files && event.target.files[0];
      if (file) {
        this.assignTextFile(file);
      }
    },
    assignAudioFile(file) {
      this.audioFile = file;
      this.audioMeta = file.name;
    },
    assignTextFile(file) {
      this.textFile = file;
      this.textMeta = file.name;
      this.removePunctuation = true;
      if (this.languageTouched) {
        return;
      }
      detectLanguageFromTextFile(file).then((detected) => {
        if (!detected || this.languageTouched) {
          return;
        }
        this.language = detected;
        this.persistLanguage(detected);
      });
    },
    handleLanguageChange() {
      this.languageTouched = true;
      this.persistLanguage(this.language);
    },
    persistLanguage(value) {
      try {
        if (!window.localStorage) {
          return;
        }
        window.localStorage.setItem("audio_to_text.language", value);
      } catch (error) {
        return;
      }
    },
    restoreLanguage() {
      try {
        if (!window.localStorage) {
          return;
        }
        const stored = window.localStorage.getItem("audio_to_text.language");
        if (!stored) {
          return;
        }
        if (stored === this.language) {
          return;
        }
        this.language = stored;
        this.languageTouched = true;
      } catch (error) {
        return;
      }
    },
    statusLabel(job) {
      const statusValue = String(job.status || "queued");
      return STATUS_LABELS[statusValue] || STATUS_LABELS.default;
    },
    statusClass(job) {
      const statusValue = String(job.status || "queued");
      return {
        "is-queued": statusValue === "queued",
        "is-running": statusValue === "running",
        "is-completed": statusValue === "completed",
        "is-failed": statusValue === "failed",
      };
    },
    jobTitle(job) {
      return job.audio_filename || UI_STRINGS.jobTitleFallback;
    },
    jobMeta(job) {
      const textName = job.text_filename || UI_STRINGS.jobTextFallback;
      const languageLabel = job.language ? job.language.toUpperCase() : "";
      const punctuation = typeof job.remove_punctuation === "boolean" ? job.remove_punctuation : null;
      const parts = [`Text: ${textName}`];
      if (languageLabel) {
        parts.push(`Lang: ${languageLabel}`);
      }
      if (punctuation !== null) {
        parts.push(punctuation ? "Punct: removed" : "Punct: kept");
      }
      return parts.join(" • ");
    },
    jobMessage(job) {
      return job.message || this.statusLabel(job);
    },
    jobProgressStyle(job) {
      const progressValue = typeof job.progress === "number" ? job.progress : 0;
      const clamped = Math.max(0, Math.min(1, progressValue));
      return `width: ${Math.round(clamped * 100)}%;`;
    },
    jobDownloadReady(job) {
      return Boolean(job.output_ready) || job.status === "completed";
    },
    downloadUrl(job) {
      return this.backend ? this.backend.downloadUrl(job.job_id) : "#";
    },
    jobDeletable(job) {
      return job.status === "completed" || job.status === "failed";
    },
    reportInvalidJob(message) {
      logError("ui.jobs.invalid_payload", new Error(message));
      this.setError(UI_STRINGS.errorInvalidJob);
    },
    buildUiIdLookup() {
      const lookup = new Map();
      for (const job of this.jobs) {
        if (!job || typeof job.job_id !== "string" || !job.job_id) {
          continue;
        }
        const uiId = typeof job.ui_id === "string" && job.ui_id ? job.ui_id : job.job_id;
        lookup.set(job.job_id, uiId);
      }
      return lookup;
    },
    normalizeIncomingJob(job, uiIdLookup) {
      if (!job || typeof job !== "object") {
        this.reportInvalidJob("job payload must be an object");
        return null;
      }
      const jobId = typeof job.job_id === "string" && job.job_id ? job.job_id : null;
      if (!jobId) {
        this.reportInvalidJob("job_id missing from payload");
        return null;
      }
      const uiId =
        uiIdLookup.get(jobId) ||
        (typeof job.ui_id === "string" && job.ui_id ? job.ui_id : jobId);
      return { ...job, job_id: jobId, ui_id: uiId, is_optimistic: false };
    },
    sortedJobs() {
      return [...this.jobs].sort((left, right) => {
        const leftOptimistic = Boolean(left.is_optimistic);
        const rightOptimistic = Boolean(right.is_optimistic);
        if (leftOptimistic !== rightOptimistic) {
          return leftOptimistic ? -1 : 1;
        }
        const leftTime = typeof left.created_at === "number" ? left.created_at : 0;
        const rightTime = typeof right.created_at === "number" ? right.created_at : 0;
        if (rightTime === leftTime) {
          return String(right.job_id || "").localeCompare(String(left.job_id || ""));
        }
        return rightTime - leftTime;
      });
    },
    createOptimisticJob() {
      const optimisticId = `local_${Date.now().toString(16)}_${Math.random().toString(16).slice(2)}`;
      const createdAt = Date.now() / 1000;
      return {
        ui_id: optimisticId,
        job_id: optimisticId,
        status: "queued",
        message: UI_STRINGS.statusQueued,
        output_ready: false,
        progress: 0,
        audio_filename: this.audioFile?.name || UI_STRINGS.jobTitleFallback,
        text_filename: this.textFile?.name || UI_STRINGS.jobTextFallback,
        language: this.language,
        remove_punctuation: this.removePunctuation,
        created_at: createdAt,
        started_at: null,
        completed_at: null,
        is_optimistic: true,
      };
    },
    applyJobsList(jobs) {
      const merged = Array.isArray(jobs) ? jobs : [];
      const uiIdLookup = this.buildUiIdLookup();
      const normalized = [];
      for (const job of merged) {
        const normalizedJob = this.normalizeIncomingJob(job, uiIdLookup);
        if (normalizedJob) {
          normalized.push(normalizedJob);
        }
      }
      const optimisticJobs = this.jobs.filter((job) => job && job.is_optimistic);
      const optimistic = optimisticJobs.filter(
        (job) => !normalized.some((entry) => entry.job_id === job.job_id)
      );
      this.jobs = [...normalized, ...optimistic];
    },
    applyJobUpdate(job) {
      const uiIdLookup = this.buildUiIdLookup();
      const normalized = this.normalizeIncomingJob(job, uiIdLookup);
      if (!normalized) {
        return;
      }
      if (this.claimPendingSubmission(normalized)) {
        return;
      }
      const index = this.jobs.findIndex((entry) => entry.job_id === normalized.job_id);
      if (index >= 0) {
        const existing = this.jobs[index];
        const uiId =
          (existing && typeof existing.ui_id === "string" && existing.ui_id) ||
          normalized.ui_id;
        this.jobs.splice(index, 1, { ...existing, ...normalized, ui_id: uiId });
        return;
      }
      this.jobs.unshift(normalized);
    },
    claimPendingSubmission(job) {
      if (!this.pendingSubmission || !job || !job.job_id) {
        return false;
      }
      if (job.job_id.startsWith("local_")) {
        return false;
      }
      if (job.audio_filename !== this.pendingSubmission.audio_filename) {
        return false;
      }
      if (job.text_filename !== this.pendingSubmission.text_filename) {
        return false;
      }
      if (job.language !== this.pendingSubmission.language) {
        return false;
      }
      if (Boolean(job.remove_punctuation) !== this.pendingSubmission.remove_punctuation) {
        return false;
      }
      const createdAt = typeof job.created_at === "number" ? job.created_at : null;
      if (createdAt !== null && Math.abs(createdAt - this.pendingSubmission.created_at_seconds) > 90) {
        return false;
      }
      const optimisticId = this.pendingSubmission.optimistic_job_id;
      const index = this.jobs.findIndex((entry) => entry.job_id === optimisticId);
      if (index >= 0) {
        const existing = this.jobs[index];
        const uiId =
          (existing && typeof existing.ui_id === "string" && existing.ui_id) ||
          optimisticId;
        this.jobs.splice(index, 1, { ...job, ui_id: uiId, is_optimistic: false });
        this.pendingSubmission = null;
        return true;
      }
      return false;
    },
    async loadJobs() {
      if (!this.backend) {
        return;
      }
      try {
        const payload = await this.backend.listJobs();
        if (Array.isArray(payload.jobs)) {
          this.applyJobsList(payload.jobs);
        }
      } catch (error) {
        logError("ui.load_jobs.failed", error);
        this.setError(UI_STRINGS.errorLoadJobs);
      }
    },
    startJobStream() {
      if (!this.backend || this.jobStream || typeof EventSource === "undefined") {
        return;
      }
      try {
        const stream = this.backend.jobStream();
        this.jobStream = stream;
        stream.addEventListener("open", () => {
          this.clearError();
        });
        stream.addEventListener("message", (event) => {
          this.clearError();
          let payload = null;
          try {
            payload = JSON.parse(event.data);
          } catch (error) {
            logError("ui.stream.parse_failed", error);
            this.setError(UI_STRINGS.errorStreamParse);
            return;
          }
          if (payload && payload.type === "keepalive") {
            return;
          }
          if (payload && Array.isArray(payload.jobs)) {
            this.applyJobsList(payload.jobs);
            return;
          }
          if (payload && payload.job_id) {
            this.applyJobUpdate(payload);
          }
        });
        stream.addEventListener("error", () => {
          this.setError(UI_STRINGS.errorStreamLost);
          if (stream.readyState !== EventSource.CLOSED) {
            return;
          }
          if (this.jobStream !== stream) {
            return;
          }
          this.jobStream = null;
          stream.close();
          if (typeof window !== "undefined") {
            window.setTimeout(() => this.startJobStream(), SSE_RECONNECT_DELAY_MS);
          }
        });
      } catch (error) {
        logError("ui.stream.init_failed", error);
        this.setError(UI_STRINGS.errorStreamInit);
      }
    },
    async startJob() {
      this.clearError();
      if (this.jobSubmitting) {
        return;
      }
      if (!this.audioFile || !this.textFile) {
        this.setError(UI_STRINGS.errorSelectFiles);
        return;
      }
      if (!this.backend) {
        this.setError(UI_STRINGS.errorUploadFailed);
        return;
      }
      this.jobSubmitting = true;
      this.setStatus(UI_STRINGS.statusQueued, UI_STRINGS.statusQueuedSub);
      const optimisticJob = this.createOptimisticJob();
      this.pendingSubmission = {
        optimistic_job_id: optimisticJob.job_id,
        audio_filename: optimisticJob.audio_filename,
        text_filename: optimisticJob.text_filename,
        language: this.language,
        remove_punctuation: this.removePunctuation,
        created_at_seconds: optimisticJob.created_at,
      };
      this.jobs.unshift(optimisticJob);
      const formData = new FormData();
      formData.append("audio", this.audioFile, this.audioFile.name);
      formData.append("text", this.textFile, this.textFile.name);
      formData.append("language", this.language);
      formData.append("remove_punctuation", this.removePunctuation ? "1" : "0");
      try {
        const payload = await this.backend.createJob(formData);
        if (payload && payload.job_id) {
          this.applyJobUpdate(payload);
        }
        this.setStatus(UI_STRINGS.statusStreaming, UI_STRINGS.statusStreamingSub);
        this.audioFile = null;
        this.textFile = null;
        this.audioMeta = UI_STRINGS.audioHint;
        this.textMeta = UI_STRINGS.textHint;
      } catch (error) {
        logError("ui.job.create_failed", error);
        this.setError(UI_STRINGS.errorUploadFailed);
      } finally {
        this.jobSubmitting = false;
      }
    },
    async deleteJob(job) {
      if (!job || !job.job_id || !this.backend) {
        return;
      }
      this.clearError();
      if (!confirm(UI_STRINGS.deleteJobConfirm)) {
        return;
      }
      try {
        await this.backend.deleteJob(job.job_id);
        await this.loadJobs();
      } catch (error) {
        logError("ui.job.delete_failed", error);
        this.setError(UI_STRINGS.errorDeleteFailed);
      }
    },
  };
}
