
from postprocess.nms import non_max_supression
from utils.cocoutils import coco2df
import json

with open("/home/stephan/Desktop/collembola_revision/stephan/second_phase/train_1500_0.6_full_plus_6bg_r50-fpn/runs/predict/exp5/result_output_mod.json","r") as f:
    res = json.load(f)

res_df = coco2df(res)
print(res_df.shape)
res_df_pro = non_max_supression(res_df)
res_df_pro_ag = non_max_supression(res_df,class_agnostic=True)
print(res_df_pro.shape)
print(res_df_pro_ag.shape)
