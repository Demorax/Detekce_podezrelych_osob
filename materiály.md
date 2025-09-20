https://huggingface.co/docs/transformers/model_doc/vitpose

https://github.com/ViTAE-Transformer/ViTPose/tree/main

https://github.com/ViTAE-Transformer/ViTPose/blob/main/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ochuman/ViTPose_huge_ochuman_256x192.py

https://github.com/jaehyunnn/ViTPose_pytorch/blob/main/configs/ViTPose_large_coco_256x192.py


https://debuggercafe.com/vitpose/

# VitPose 
- Using YOLOv3 human detector. Note the configs here are only for evaluation.
- https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBaW1CZ1lWN0pqVGxnUy1vQXZFVjRNVEQtLVhyP2U9RWVXMkZ1&cid=E534267B85818129&id=E534267B85818129%21175&parId=E534267B85818129%21162&o=OneUp
## Zmeny
v vit.py

zakomentovano

super().init_weights(pretrained, patch_padding=self.patch_padding)