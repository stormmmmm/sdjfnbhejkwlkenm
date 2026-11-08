# Синтез ДОЭ

## Подготовка входных данных

В корне проекта создайте `input/`. Для каждого комплекта создайте в ней
отдельную папку, например `input/set_a/`, и поместите туда четыре
8-битных PNG в градациях серого (режим `L`), каждый размером 128×128 пикселей.
Полностью чёрные изображения не принимаются.

| Файл | Расстояние восстановления |
| --- | ---: |
| `1.png` | 24,1 мм |
| `2.png` | 25,3 мм |
| `3.png` | 26,5 мм |
| `4.png` | 27,7 мм |

## Сборка и запуск в Docker

Выполните команды из корня проекта. Образ использует CPU и включает обученные
веса `weights/model.keras`.

**Bash**

```bash
docker build -t case-mephi .
mkdir -p output/docker
docker run --rm --network none \
  --mount "type=bind,source=$PWD/input,target=/data/input,readonly" \
  --mount "type=bind,source=$PWD/output/docker,target=/data/output" \
  case-mephi
```

**PowerShell**

```powershell
docker build -t case-mephi .
New-Item -ItemType Directory -Force output/docker | Out-Null
$inputPath = (Resolve-Path input).Path
$outputPath = (Resolve-Path output/docker).Path
docker run --rm --network none `
  --mount "type=bind,source=$inputPath,target=/data/input,readonly" `
  --mount "type=bind,source=$outputPath,target=/data/output" `
  case-mephi
```

## Локальный запуск в Windows

Установите Python 3.14 и выполните в PowerShell из корня проекта.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --model weights/model.keras
```

## Результаты

Docker записывает результаты в `output/docker/`, локальный запуск — в `output/`.
В папке каждого комплекта находятся один `doe.png`, четыре реконструкции
`1.png`–`4.png`, `metrics.json` с Pearson и SSIM по плоскостям и `run.json`
со временем синтеза. Общая сводка сохраняется в `summary.json`.

Для повторного запуска выберите новый каталог результатов. В Docker
измените путь монтирования `/data/output` на стороне хоста, при локальном
запуске укажите, например, `--output output/repeat`.
