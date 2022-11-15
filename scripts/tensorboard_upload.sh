#!/bin/bash
SCRIPT_DIR=`dirname $0`
cd $SCRIPT_DIR
#rootへ移動
cd ../
# これ以下に処理を書く
tensorboard dev upload \
--logdir ./lightning_logs \
--name "(optional) My latest experiment" \
--description "(optional) Simple comparison of several hyperparameters" \
--one_shot

pwd