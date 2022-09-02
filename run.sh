for p_lr in 0.001 0.005 0.01
do
	for f_lr in 0.001 0.005 0.01
	do
	  for dropout in 0.0
	  do
	    for pretrain_wd in 1e-4
      do
        python train-clf.py \
        --pretrain-lr $p_lr \
        --finetune-lr $f_lr \
        --dropout $dropout \
        --pretrain-wd $pretrain_wd
      done
    done
  done
done
