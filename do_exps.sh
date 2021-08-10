for hdim in 8 12 16
do
    for edim in 64 32 24 16 8 4 2
    do
        python main.py --cuda --hid_dim $hdim --embed_dim $edim --lr 0.001 --epochs 880 --print_update 5000
    done
done
