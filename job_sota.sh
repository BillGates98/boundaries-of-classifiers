# python3 ./sota.py --input_path ./data/ --output_path ./outputs/ --suffix abt-buy
# python3 ./sota.py --input_path ./data/feature_vector/ --output_path ./outputs/ --suffix amazon-google
# python3 ./sota.py --input_path ./data/feature_vector/ --output_path ./outputs/ --suffix cosmetics
# python3 ./sota.py --input_path ./data/ --output_path ./outputs/ --suffix wdc_xlarge_shoes


# python3 ./sota.py --input_path ./data/ --output_path ./outputs/ --suffix doremus-100
# python3 ./sota.py --input_path ./data/ --output_path ./outputs/ --suffix doremus-200
# python3 ./sota.py --input_path ./data/ --output_path ./outputs/ --suffix doremus-300
# python3 ./sota.py --input_path ./data/ --output_path ./outputs/ --suffix doremus-400
# python3 ./sota.py --input_path ./data/ --output_path ./outputs/ --suffix doremus-500

# python3 ./sota.py --input_path ./data/ --output_path ./outputs/ --suffix anatomy-10
# python3 ./sota.py --input_path ./data/ --output_path ./outputs/ --suffix anatomy-20
# python3 ./sota.py --input_path ./data/ --output_path ./outputs/ --suffix anatomy-30
# python3 ./sota.py --input_path ./data/ --output_path ./outputs/ --suffix anatomy-40
# python3 ./sota.py --input_path ./data/ --output_path ./outputs/ --suffix anatomy-50

for dataset in 'anatomy' 'doremus' 'SPIMBENCH_large-2016' 'SPIMBENCH_small-2019' 'UOBM_small-2016'
# for dataset in 'SPIMBENCH_large-2016' 'SPIMBENCH_small-2019'
do
    for dim in 10 20 30 40 50
    do
        python3 ./sota.py --input_path ./data/word2vec/ --output_path ./outputs/ --suffix /$dataset-$dim
        # exit
    done
done

# python3 ./sota.py --input_path ./data/node2vec/ --output_path ./outputs/node2vec/ --suffix /doremus-10
# python3 ./sota.py --input_path ./data/node2vec/ --output_path ./outputs/node2vec/ --suffix /doremus-20
# python3 ./sota.py --input_path ./data/node2vec/ --output_path ./outputs/node2vec/ --suffix /doremus-30
# python3 ./sota.py --input_path ./data/node2vec/ --output_path ./outputs/node2vec/ --suffix /doremus-40
# python3 ./sota.py --input_path ./data/node2vec/ --output_path ./outputs/node2vec/ --suffix /doremus-50
