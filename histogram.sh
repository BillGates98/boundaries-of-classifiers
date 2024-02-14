# python3 ./histogram.py --input_path ./outputs/ --output_path ./outputs/ --suffix anatomy

for dataset in 'anatomy' 'doremus' 'SPIMBENCH_large-2016' 'SPIMBENCH_small-2019' 'UOBM_small-2016'
do
    python3 ./histogram.py --input_path ./outputs/ --output_path ./outputs/ --suffix $dataset
done