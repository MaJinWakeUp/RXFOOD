import os
import cv2
import py_sod_metrics
import neptune.new as neptune

# run = neptune.init_run(
#         project="majinwakeup/CrossAttention",
#         api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NzIzZmMyYS0zNTZkLTRmMjQtOTI5Ny05YWJjMGM0OTMzNzEifQ==",
#     )  # your credentials

FM = py_sod_metrics.Fmeasure()
WFM = py_sod_metrics.WeightedFmeasure()
SM = py_sod_metrics.Smeasure()
EM = py_sod_metrics.Emeasure()
MAE = py_sod_metrics.MAE()

# sample_binary = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=True)
# overall_binary = dict(
#     with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=False
# )
# FMv2 = py_sod_metrics.FmeasureV2(
#     metric_handlers={
#         # 灰度数据指标
#         "fm": py_sod_metrics.FmeasureHandler(with_adaptive=True, with_dynamic=True, beta=0.3),
#         "f1": py_sod_metrics.FmeasureHandler(with_adaptive=True, with_dynamic=True, beta=0.1),
#         "pre": py_sod_metrics.PrecisionHandler(with_adaptive=True, with_dynamic=True),
#         "rec": py_sod_metrics.RecallHandler(with_adaptive=True, with_dynamic=True),
#         "iou": py_sod_metrics.IOUHandler(with_adaptive=True, with_dynamic=True),
#         "dice": py_sod_metrics.DICEHandler(with_adaptive=True, with_dynamic=True),
#         "spec": py_sod_metrics.SpecificityHandler(with_adaptive=True, with_dynamic=True),
#         "ber": py_sod_metrics.BERHandler(with_adaptive=True, with_dynamic=True),
#         # 二值化数据指标的特殊情况一：各个样本独立计算指标后取平均
#         "sample_bifm": py_sod_metrics.FmeasureHandler(**sample_binary, beta=0.3),
#         "sample_bif1": py_sod_metrics.FmeasureHandler(**sample_binary, beta=1),
#         "sample_bipre": py_sod_metrics.PrecisionHandler(**sample_binary),
#         "sample_birec": py_sod_metrics.RecallHandler(**sample_binary),
#         "sample_biiou": py_sod_metrics.IOUHandler(**sample_binary),
#         "sample_bidice": py_sod_metrics.DICEHandler(**sample_binary),
#         "sample_bispec": py_sod_metrics.SpecificityHandler(**sample_binary),
#         "sample_biber": py_sod_metrics.BERHandler(**sample_binary),
#         # 二值化数据指标的特殊情况二：汇总所有样本的tp、fp、tn、fn后整体计算指标
#         "overall_bifm": py_sod_metrics.FmeasureHandler(**overall_binary, beta=0.3),
#         "overall_bif1": py_sod_metrics.FmeasureHandler(**overall_binary, beta=1),
#         "overall_bipre": py_sod_metrics.PrecisionHandler(**overall_binary),
#         "overall_birec": py_sod_metrics.RecallHandler(**overall_binary),
#         "overall_biiou": py_sod_metrics.IOUHandler(**overall_binary),
#         "overall_bidice": py_sod_metrics.DICEHandler(**overall_binary),
#         "overall_bispec": py_sod_metrics.SpecificityHandler(**overall_binary),
#         "overall_biber": py_sod_metrics.BERHandler(**overall_binary),
#     }
# )

model_name = 'S2MAaddAttv3'
# model_name = 'DualFPNwithMCA_manipulation'
data_root = '/home/majin/projects/CrossAttention/results/RDSOD'
mask_root = os.path.join(data_root, 'gt')
pred_root = os.path.join(data_root, model_name)
mask_name_list = sorted(os.listdir(mask_root))

# nep_params = {
#         "task": model_name.split('_')[-1],
#         "Backbone": model_name.split('with')[0][4:],
#         "attention": model_name.split('with')[1].split('_')[0],
#         "learning_rate": 2e-5, 
#         "optimizer": "Adam",
#         "lr_scheduler": "DynamicLR",
#         "alpha": 'learnable',
#         }
# run["parameters"] = nep_params

for i, mask_name in enumerate(mask_name_list):
    print(f"[{i}] Processing {mask_name}...")
    mask_path = os.path.join(mask_root, mask_name)
    # pred_path = os.path.join(pred_root, mask_name.replace('_gt.png', '.png'))
    pred_path = os.path.join(pred_root, mask_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    if mask.shape != pred.shape:
        print(f"Warning: {mask_name} shape not match!")
        # pred = cv2.resize(pred, mask.shape[::-1])
        continue
    FM.step(pred=pred, gt=mask)
    WFM.step(pred=pred, gt=mask)
    SM.step(pred=pred, gt=mask)
    EM.step(pred=pred, gt=mask)
    MAE.step(pred=pred, gt=mask)
    # FMv2.step(pred=pred, gt=mask)

fm = FM.get_results()["fm"]
wfm = WFM.get_results()["wfm"]
sm = SM.get_results()["sm"]
em = EM.get_results()["em"]
mae = MAE.get_results()["mae"]
# run["test/S-measure"] = sm
# run["test/E-measure"] = em["adp"]
# run["test/F-measure"] = fm["adp"]
# run["test/MaxF"] = fm["curve"].max()
# run["test/MAE"] = mae

print(f"S-measure: {sm:.4f}, E-measure: {em['adp']:.4f}, F-measure: {fm['adp']:.4f}, MaxF: {fm['curve'].max():.4f}, MAE: {mae:.4f}")
# fmv2 = FMv2.get_results()
# run.stop()
