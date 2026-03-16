#!/bin/bash
# copy_subagent_models.sh
# 将子代理训练好的模型复制并重命名到超代理读取的目录
# Usage: ./scripts/copy_subagent_models.sh <dataset> <action_mode> <exp> <alpha> \
#        <slope1_epoch> <slope2_epoch> <slope3_epoch> \
#        <vol1_epoch>   <vol2_epoch>   <vol3_epoch>
#
# Example:
#   ./scripts/copy_subagent_models.sh MRMB short short_action_5_1m 1 7 10 2 11 12 11     vol=11,vol2=12,vol3=11,slope1=7,slope2=10,slope3=2
#   ./scripts/copy_subagent_models.sh MRMB short short_action_2_1m 1 11 2 13 7 14 11     slope1=11,slope2=2,slope3=13,vol1=7,vol2=14,vol3=11,
#   ./scripts/copy_subagent_models.sh MRMB long long_action_2_1m 1 7 1 13 4 11 13        slope1=7,slope2=1,slope3=13,vol1=4,vol2=11,vol3=13,
#   ./scripts/copy_subagent_models.sh MRMB short short_action_2_30s 1 11 13 13 13 14 12  slope1=11,slope2=13,slope3=13,vol1=13,vol2=14,vol3=12
#   ./scripts/copy_subagent_models.sh MRMB short short_action_5_30s 1 13 14 13 14 14 14  slope1=13,slope2=14,slope3=13,vol1=14,vol2=14,vol3=14
#   ./scripts/copy_subagent_models.sh MRMB both both_action_2_30s 1 14 11 9 12 11 10     slope1=14,slope2=11,slope3=9,vol1=12,vol2=11,vol3=10
#   ./scripts/copy_subagent_models.sh MRMB both both_action_5_30s 1 11 10 9 12 13 10     slope1=11,slope2=10,slope3=9,vol1=12,vol2=13,vol3=10
#   ./scripts/copy_subagent_models.sh MRMB short short_action_2_10s 1 11 10 13 13 12 12     slope1=11,slope2=10,slope3=13,vol1=13,vol2=12,vol3=12




set -e

if [ "$#" -ne 10 ]; then
    echo "Usage: $0 <dataset> <action_mode> <exp> <alpha> \\"
    echo "          <slope1_epoch> <slope2_epoch> <slope3_epoch> \\"
    echo "          <vol1_epoch>   <vol2_epoch>   <vol3_epoch>"
    exit 1
fi

DATASET=$1
ACTION_MODE=$2
EXP=$3
ALPHA=$4
SLOPE1_EPOCH=$5
SLOPE2_EPOCH=$6
SLOPE3_EPOCH=$7
VOL1_EPOCH=$8
VOL2_EPOCH=$9
VOL3_EPOCH=${10}

SEED=6234571

# Source base: result/low_level/{dataset}/{exp}/{clf}/label_{n}/{alpha}/seed_{seed}/epoch_{N}/trained_model.pkl
# Dest  base: result/low_level/{dataset}/{action_mode}/{exp}/best_model/{clf}/{n}/best_model.pkl

copy_model() {
    local CLF=$1
    local LABEL_N=$2
    local EPOCH=$3

    SRC="./result/low_level/${DATASET}/${EXP}/${CLF}/label_${LABEL_N}/${ALPHA}/seed_${SEED}/epoch_${EPOCH}/trained_model.pkl"
    DST_DIR="./result/low_level/${DATASET}/${ACTION_MODE}/${EXP}/best_model/${CLF}/${LABEL_N}"
    DST="${DST_DIR}/best_model.pkl"

    if [ ! -f "${SRC}" ]; then
        echo "[ERROR] Source not found: ${SRC}"
        exit 1
    fi

    mkdir -p "${DST_DIR}"
    rm -rf "${DST}"
    echo "[OK] rm -rf ${DST}"
    cp "${SRC}" "${DST}"
    echo "[OK] ${SRC} -> ${DST}"
}

echo "=== Copying sub-agent models ==="
echo "  dataset     : ${DATASET}"
echo "  action_mode : ${ACTION_MODE}"
echo "  exp         : ${EXP}"
echo "  alpha       : ${ALPHA}"
echo "  slope epochs: 1=${SLOPE1_EPOCH}, 2=${SLOPE2_EPOCH}, 3=${SLOPE3_EPOCH}"
echo "  vol   epochs: 1=${VOL1_EPOCH},   2=${VOL2_EPOCH},   3=${VOL3_EPOCH}"
echo ""

copy_model slope 1 ${SLOPE1_EPOCH}
copy_model slope 2 ${SLOPE2_EPOCH}
copy_model slope 3 ${SLOPE3_EPOCH}

copy_model vol 1 ${VOL1_EPOCH}
copy_model vol 2 ${VOL2_EPOCH}
copy_model vol 3 ${VOL3_EPOCH}

echo ""
echo "=== Done. All 6 models copied. ==="
