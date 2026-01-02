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

# Super Resolution
- https://github.com/aswintechguy/Deep-Learning-Projects/tree/main/Super%20Resolution%20-%20OpenCV
- nebo https://github.com/Saafke/EDSR_Tensorflow/tree/master/models

## Opencv s cudou
:: 1) Remove CPU wheels if present
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless

:: 2) Make sure the package folder 'cv2\' is gone so Python imports the .pyd
set "SITE=%CONDA_PREFIX%\Lib\site-packages"
rmdir /s /q "%SITE%\cv2" 2>nul

:: 3) Ensure your CUDA .pyd is installed at top-level as cv2.pyd
:: (replace BUILD path with your actual build dir if needed)
set "BUILD=C:\opencv-install\4.12.0"
copy /Y "%BUILD%\lib\python3\Release\cv2.cp39-win_amd64.pyd" "%SITE%\cv2.pyd"

:: 4) Copy the OpenCV + CUDA/cuDNN DLLs (you already staged most—re-run is harmless)
set "CUDA12=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin"
xcopy /Y /I "%BUILD%\bin\Release\opencv_*.dll" "%SITE%\"
xcopy /Y /I "%BUILD%\bin\Release\*ffmpeg*.dll" "%SITE%\"  2>nul
for %F in (cudart64_12.dll cublas64_12.dll cublasLt64_12.dll nvjpeg64_12.dll cufft64_11.dll nvrtc64_120_0.dll) do @if exist "%CUDA12%\%F" copy /Y "%CUDA12%\%F" "%SITE%\"
xcopy /Y /I "%CUDA12%\npp*.dll" "%SITE%\"
xcopy /Y /I "C:\Program Files\cudnn-windows-x86_64-8.9.7.29_cuda12-archive\bin\cudnn*.dll" "%SITE%\"

:: 5) Sanity check loaders
"%CONDA_PREFIX%\python.exe" -c "import os,ctypes; d=r'%SITE%'; ctypes.WinDLL(os.path.join(d,'opencv_world4120.dll')); ctypes.WinDLL(os.path.join(d,'cv2.pyd')); print('cv2.pyd + world4120 load OK')"

:: 6) Verify you’re importing the right cv2 now (should show cv2.pyd, CUDA True, device > 0)
"%CONDA_PREFIX%\python.exe" -c "import cv2, numpy as np; print('cv2 file:', getattr(cv2,'__file__','<none>')); print('cv2 version:', getattr(cv2,'__version__','<no>')); print('CUDA?', 'CUDA' in cv2.getBuildInformation()); print('devices=', cv2.cuda.getCudaEnabledDeviceCount())"

# RT-Detr v2
- https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch