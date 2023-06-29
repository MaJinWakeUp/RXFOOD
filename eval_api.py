import os
import cv2
import py_sod_metrics


FM = py_sod_metrics.Fmeasure()
WFM = py_sod_metrics.WeightedFmeasure()
SM = py_sod_metrics.Smeasure()
EM = py_sod_metrics.Emeasure()
MAE = py_sod_metrics.MAE()

model_name = 'DualCPDwithRXFOOD_MSSOD'
# model_name = 'DualFPNwithMCA_manipulation'
data_root = './results/MSSOD'
mask_root = os.path.join(data_root, 'gt')
pred_root = os.path.join(data_root, model_name)
mask_name_list = sorted(os.listdir(mask_root))

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
