cd ../
seed=5
for mod in Eq1to2Combo  BaselineEmbedDeepSets #Eq1to2Set
do
    for wd in 0 0.001
    do
        for hdim in 8 16
        do
            for edim in 32 64
            do
                python main_mask.py --cuda --hid_dim $hdim --embed_dim $edim --lr 0.0002 --epochs 880 --print_update 2000 --model $mod --seed $seed --weight_decay $wd
            done
        done
    done
done
