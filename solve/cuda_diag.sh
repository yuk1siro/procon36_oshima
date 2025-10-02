#!/usr/bin/env bash
set -e
if ! command -v nvidia-smi >/dev/null; then
  echo "❌ WSLからGPUが見えていません（NVIDIAドライバ未対応 or WSL設定）"
  exit 1
fi
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
DRV_CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]\+\).*/\1/p' | head -n1)
TK=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]\+\),.*/\1/p')
echo "Driver version   : $DRV"
echo "Driver supports  : CUDA $DRV_CUDA"
echo "Toolkit (nvcc)   : $TK"
if [ -n "$TK" ] && [ -n "$DRV_CUDA" ]; then
  awk -v tk="$TK" -v dr="$DRV_CUDA" 'BEGIN{
    split(tk,a,"."); split(dr,b,".");
    if (a[1]>b[1] || (a[1]==b[1] && a[2]>b[2])) {
      print "❌ ツールキットがドライバ対応より新しいです。ドライバ更新 か cuda-toolkit-" dr " を入れてください。";
      exit 1
    } else {
      print "✅ 版の整合性は概ねOKです。";
    }
  }'
fi
