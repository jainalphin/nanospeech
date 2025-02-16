#!/bin/bash

INPUT_DIR="extracted/LibriTTS_R"

convert_file() {
  infile="$1"
  outfile="${infile%.wav}.mp3"
  ffmpeg -i $infile -ab 64k $outfile -y &>/dev/null
}

export -f convert_file

find "$INPUT_DIR" -type f -name "*.wav" | parallel -j "$(sysctl -n hw.ncpu)" convert_file
