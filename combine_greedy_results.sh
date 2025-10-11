#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 NAME" >&2
  exit 1
fi

name=$1

# Expand only matching files; otherwise leave array empty (no literal pattern)
shopt -s nullglob
files=( "${name}"[0-9]*_*.txt )

if (( ${#files[@]} == 0 )); then
  echo "No files match pattern: ${name}<NUM1>_<NUM2>.txt" >&2
  exit 1
fi

# Collect pairs: "<NUM1>\t<filepath>"
pairs=()
for f in "${files[@]}"; do
  if [[ $f =~ ^${name}([0-9]+)_.+\.txt$ ]]; then
    pairs+=( "${BASH_REMATCH[1]}"$'\t'"$f" )
  fi
done

if (( ${#pairs[@]} == 0 )); then
  echo "No files matched the exact pattern ${name}<NUM1>_<NUM2>.txt" >&2
  exit 1
fi

# Sort by NUM1 numerically and concatenate
printf '%s\0' "${pairs[@]}" \
  | sort -z -n -t $'\t' -k1,1 \
  | cut -z -f2 \
  | xargs -0 cat -- > "${name}.txt"

echo "Wrote ${name}.txt"