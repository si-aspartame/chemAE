#!/bin/bash
for n in $(seq 2 3)
do
	for i in raw_input_and_external_dm mds umap pca vae 
	do
			echo --------------------$i LD:$n
			python3 toy.py --data 'roll' --model $i --z_dim $n
	done
done