#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64g
#SBATCH -J AUG_MATCH
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH -o ./aug_match_all_roberta_%j.out
#SBATCH -e ./aug_match_all_roberta_%j.err

set -eo pipefail
module load python/3.11.12/7vjddo3
source /home/nbpulsone/sandbox_research/rodoem_llm_ft/modenv/bin/activate

STUDENT_LM="roberta"
TEACHER_LM="Qwen/Qwen2.5-7B-Instruct"
STRATEGY="uncertain"
C_EFF=0.20
RUN_ID=0

COMMON=(
    --student_lm "$STUDENT_LM"
    --run_id "$RUN_ID"
    --batch_size 64 \
    --max_len 128 \
    --n_epochs 40 \
    --lr 3e-5 \
    --logdir "checkpoints/aug_${SLURM_JOB_ID}"
)

AUG_ARGS=(
    --teacher_lm "$TEACHER_LM"
    --active_aug
    --selection_strategy "$STRATEGY"
    --c_eff "$C_EFF"
    --warmup_epochs 3
    --active_interval 3
    --active_rounds 5
)

if [[ "$STRATEGY" == baseline ]]; then
    AUG_ARGS=()
fi

for task in \
    Structured/Fodors-Zagats \
    Structured/Amazon-Google \
    Textual/Abt-Buy \
    Dirty/DBLP-ACM \
    Dirty/iTunes-Amazon \
    Dirty/Walmart-Amazon; do
    echo "Task=$task Student=$STUDENT_LM Strategy=$STRATEGY C_eff=$C_EFF Seed=$RUN_ID"
    
    python -u train_ditto.py --task "$task" "${COMMON[@]}" "${AUG_ARGS[@]}"

done
