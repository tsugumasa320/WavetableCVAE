#!/bin/bash
SCRIPT_DIR=`dirname $0`
cd $SCRIPT_DIR
#rootへ移動
cd ../
# これ以下に処理を書く
tensorboard --logdir=lightning_logs/

pwd

