gs=(0.2 -0.2 0.7 -0.7 0.9 -0.9)
r_starts=("0.2" "0.5" "0.8")
r_ends=("0.2+1e-3" "0.5+1e-3" "0.8+1e-3")
alpha_starts=("0" "torch.pi / 6" "torch.pi / 2" "torch.pi * 2 / 3" "torch.pi * 5 / 6")
alpha_ends=("1e-2" "torch.pi / 6 + 1e-2" "torch.pi / 2 + 1e-2" "torch.pi * 2 / 3 + 1e-2" "torch.pi * 5 / 6 + 1e-2")

echo "" &> log.txt

for((k=0;k<5;k++)); do
    echo "-------------------- alpha_start = '$alpha_start' ----------------------" >> log.txt
    alpha_start=${alpha_starts[k]}
    alpha_end=${alpha_ends[k]}
for((i=0;i<3;i++)); do
    r_start=${r_starts[i]}
    r_end=${r_ends[i]}
    for g in ${gs[@]}; do
        echo "Processing: g = $g, alpha_start = '$alpha_start', r_start = $r_start ..." >> log.txt
        python sample_visualize.py --config ./configs/guide.conf --g $g --alpha_start $alpha_start --alpha_end $alpha_end \
                    --r_start $r_start --r_end $r_end
    done
done
done