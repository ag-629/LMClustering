#!/bin/bash

#Example of the pipeline
#Adjust file paths and add langauges as needed

#vector creation is in the 'scripts' directory

#train.py and confidence.py are in the 'training' directory

#Be sure to differentialte between forward and reverse for file outputs

for lang in Maltese; do
    python3 create_word_vectors.py ${lang}.bible.txt ${lang} ${lang}.bin
    python3 convert_to_vec.py ${lang}.bin ${lang} ${lang}.vec
    python3 train.py --train_data_path ${lang}.bible.txt --lang ${lang} --num_epochs 100 --learning_rate 0.001 --batch_size 16 --embedding_size 128 --hidden_size 128
    python3 confidences.py ${lang}.bible.txt ${lang}
    python3 cluster.py -C -S 0 -l --no-N --no-D ./${lang}.vec ./${lang}_Confidences ${lang} ./${lang}_Clusters
done
