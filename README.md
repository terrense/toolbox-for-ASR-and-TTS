# toolbox-for-ASR-and-TTS

A lightweight toolbox to run **ASR** (Automatic Speech Recognition) and **TTS** (Text-to-Speech) locally using **Docker Compose**.  
This repo provides **two separate Docker Compose files**: one for ASR and one for TTS. You can start them as two containers on your machine.

---

## Features

- ✅ ASR service via Docker Compose
- ✅ TTS service via Docker Compose
- ✅ Run locally with two containers
- ✅ Model names are listed in `services/models/damo/`
- 🔽 Model files should be downloaded from **ModelScope** (not included in this repo)

---

## Repository Structure (high level)

```text
.
├── docker-compose.asr.yml      # ASR compose file (name may vary)
├── docker-compose.tts.yml      # TTS compose file (name may vary)
└── services/
    └── models/
        └── damo/               # model names / directory placeholders
