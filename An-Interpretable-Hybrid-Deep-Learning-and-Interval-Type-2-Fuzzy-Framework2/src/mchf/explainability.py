from __future__ import annotations
from pathlib import Path
import numpy as np
import torch


def gradcam_pp(effnet_cbam_model, image_tensor, target_category=1):
    """Grad-CAM++ for the final EfficientNetV2+CBAM feature map.

    Requires the `grad-cam` package. The provided model should expose
    `backbone` and `cbam`; the target layer is the last backbone stage.
    """
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except Exception as e:
        raise ImportError('Install grad-cam>=1.5 for Grad-CAM++') from e
    # CAM package expects a classifier. Users should pass the trained branch
    # classifier wrapper; target_layers can be overridden if needed.
    target_layers=[list(effnet_cbam_model.modules())[-2]]
    cam=GradCAMPlusPlus(model=effnet_cbam_model,target_layers=target_layers)
    return cam(input_tensor=image_tensor,targets=[ClassifierOutputTarget(target_category)])[0]


def shap_summary(model_predict_fn, background, samples, feature_names, out_path):
    """Generate SHAP summary plot for fused/tabular features."""
    try: import shap
    except Exception as e: raise ImportError('Install shap>=0.45') from e
    explainer=shap.Explainer(model_predict_fn, background, feature_names=feature_names)
    values=explainer(samples)
    import matplotlib.pyplot as plt
    shap.summary_plot(values, samples, feature_names=feature_names, show=False)
    Path(out_path).parent.mkdir(parents=True,exist_ok=True); plt.tight_layout(); plt.savefig(out_path,dpi=300,bbox_inches='tight'); plt.close()
    return values
