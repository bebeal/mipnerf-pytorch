#!/bin/bash
mkdir -p data
cd data
echo "Getting LLFF Dataset"
wget -q --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=16VnMcF1KJYxN9QId6TClMsZRahHNMW5g' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=16VnMcF1KJYxN9QId6TClMsZRahHNMW5g" -O out.zip && rm -rf /tmp/cookies.txt
unzip -q out.zip
rm -rf out.zip
