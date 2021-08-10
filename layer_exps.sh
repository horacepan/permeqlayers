seed=2

for mod in Eq1to2Set
do
    for wd in 0.001 0.01 0.1
    do
        for hdim in 16
        do
            for edim in 32
            do
                python main_mask.py --cuda --hid_dim $hdim --embed_dim $edim --lr 0.0001 --epochs 880 --print_update 5000 --model $mod --seed $seed --weight_decay $wd
            done
        done
    done
done
