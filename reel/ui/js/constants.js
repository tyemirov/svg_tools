// @ts-check

export const UI_STRINGS = Object.freeze({
  badge: "Audio to Text",
  title: "Forced alignment, ready for subtitles",
  description: "Drop audio or video plus your transcript. A background job aligns each word and builds an SRT you can download.",
  audioTitle: "Audio or video file",
  audioHint: "Drop a file or click to browse",
  textTitle: "Transcript text file",
  textHint: "Drop a file or click to browse",
  languageLabel: "Language",
  punctuationLabel: "Remove punctuation",
  punctuationEnabled: "Enabled",
  runButton: "Align and Build SRT",
  statusReady: "Ready to align.",
  statusSub: "Upload files to begin.",
  jobsHeader: "Session jobs",
  downloadLabel: "Download SRT",
  deleteJobTitle: "Delete finished job",
  deleteJobConfirm: "Delete this job?",
  statusQueued: "Queued.",
  statusQueuedSub: "Uploading files and preparing alignment.",
  statusStreaming: "Aligning.",
  statusStreamingSub: "Streaming job updates.",
  errorSelectFiles: "Select both an audio file and a transcript file.",
  errorDeleteFailed: "Failed to delete the job.",
  errorLoadJobs: "Failed to load job history.",
  errorUploadFailed: "Failed to start alignment.",
  errorInvalidJob: "Received invalid job data.",
  errorStreamParse: "Failed to parse job updates.",
  errorStreamLost: "Connection lost while streaming job updates.",
  errorStreamInit: "Failed to connect to job updates.",
  jobTitleFallback: "Audio alignment",
  jobTextFallback: "unknown text",
});

export const STATUS_LABELS = Object.freeze({
  queued: "Queued",
  running: "Running",
  completed: "Complete",
  failed: "Failed",
  default: "Unknown",
});

export const DEFAULT_LANGUAGE = "en";
export const DEFAULT_REMOVE_PUNCTUATION = true;
export const DEFAULT_BACKEND_URL = "http://localhost:8080";
export const SSE_RECONNECT_DELAY_MS = 2000;

export const AUDIO_INPUT_ACCEPT =
  "audio/*,video/*,.wav,.wave,.mp3,.m4a,.aac,.flac,.ogg,.mp4,.mov,.m4v";
export const TEXT_INPUT_ACCEPT = ".txt,.md,.srt,.sbv";

export const LANGUAGE_OPTIONS = Object.freeze([
  { value: "en", label: "English (en)" },
  { value: "fr", label: "French (fr)" },
  { value: "de", label: "German (de)" },
  { value: "es", label: "Spanish (es)" },
  { value: "it", label: "Italian (it)" },
  { value: "ja", label: "Japanese (ja)" },
  { value: "zh", label: "Chinese (zh)" },
  { value: "nl", label: "Dutch (nl)" },
  { value: "uk", label: "Ukrainian (uk)" },
  { value: "pt", label: "Portuguese (pt)" },
  { value: "ar", label: "Arabic (ar)" },
  { value: "cs", label: "Czech (cs)" },
  { value: "ru", label: "Russian (ru)" },
  { value: "pl", label: "Polish (pl)" },
  { value: "hu", label: "Hungarian (hu)" },
  { value: "fi", label: "Finnish (fi)" },
  { value: "fa", label: "Persian (fa)" },
  { value: "el", label: "Greek (el)" },
  { value: "tr", label: "Turkish (tr)" },
  { value: "da", label: "Danish (da)" },
  { value: "he", label: "Hebrew (he)" },
  { value: "vi", label: "Vietnamese (vi)" },
  { value: "ko", label: "Korean (ko)" },
  { value: "ur", label: "Urdu (ur)" },
  { value: "te", label: "Telugu (te)" },
  { value: "hi", label: "Hindi (hi)" },
  { value: "ca", label: "Catalan (ca)" },
  { value: "ml", label: "Malayalam (ml)" },
  { value: "no", label: "Norwegian Bokmal (no)" },
  { value: "nn", label: "Norwegian Nynorsk (nn)" },
  { value: "sk", label: "Slovak (sk)" },
  { value: "sl", label: "Slovenian (sl)" },
  { value: "hr", label: "Croatian (hr)" },
  { value: "ro", label: "Romanian (ro)" },
  { value: "eu", label: "Basque (eu)" },
  { value: "gl", label: "Galician (gl)" },
  { value: "ka", label: "Georgian (ka)" },
]);
