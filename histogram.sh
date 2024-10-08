# python3 ./histogram.py --input_path ./outputs/ --output_path ./outputs/ --suffix anatomy

for dataset in 'anatomy' 'doremus' 'SPIMBENCH_large-2016' 'SPIMBENCH_small-2019' 'UOBM_small-2016'
do
    for model in 'r2v' 'RESCAL' 'TransE' 'DistMult'
    do
        # mkdir -p ./outputs/$model/$dataset
        python3.8 ./histogram.py --input_path ./outputs_balanced/$model/ --output_path ./outputs_balanced/$model/ --suffix $dataset --model $model
        python3.8 ./histogram.py --input_path ./outputs_unbalanced/$model/ --output_path ./outputs_unbalanced/$model/ --suffix $dataset --model $model
    done
    # python3.8 ./histogram.py --input_path ./outputs/ --output_path ./outputs/ --suffix $dataset
done