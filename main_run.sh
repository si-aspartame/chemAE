#!/bin/bash
for n in $(seq 2 10)
do
	for i in mds #umap pca vae 
	do
			echo --------------------$i LD:$n
			python3 main.py --data 'qm7' --model $i --z_dim $n
	done
done