cd ../
model=BaselineEmbedDeepSets
for hdim in 2 4 8 16
do
    for edim in 64 32
    do
        python main_mask.py --model $model --cuda --hid_dim $hdim --embed_dim $edim --lr 0.001 --epochs 300 --print_update 1000
    done
done
