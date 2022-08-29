python v1-train.py \
--pretrain-epochs \
200 \
--finetune-epochs \
500 \
--pretrain-lr \
0.001 \
--finetune-lr \
0.001 \
--finetune-interval \
2 \
--node-classification \
--decoder-type \
inner_product \
--node-loss-type \
bce_loss \
--attr-loss-type \
bce_loss