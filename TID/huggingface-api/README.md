---
title: TID İşaret Dili API
emoji: 🤟
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# TID İşaret Dili API

Türk İşaret Dili algılama projesi için backend API.

## Modeller
- **Random Forest** — Harf tanıma (23 harf, %95 doğruluk)
- **1D CNN** — Kelime tanıma (70 kelime)

## Endpointler
- `GET /` — Sağlık kontrolü
- `POST /predict/rf` — Random Forest tahmini
- `POST /predict/cnn1d` — 1D CNN tahmini
