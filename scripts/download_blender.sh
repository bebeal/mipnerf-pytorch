#!/bin/bash
mkdir -p data
cd data
echo "Getting Blender Dataset"
wget -q --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG" -O out.zip && rm -rf /tmp/cookies.txt
unzip -q out.zip
rm -rf out.zip
