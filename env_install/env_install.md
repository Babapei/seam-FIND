## 显卡驱动安装

https://www.nvidia.com/en-us/drivers/

显卡太垃圾了，版本太低了，在这台电脑直接用cpu跑

## python安装

https://www.python.org/downloads/windows/

选择3.10

![替代文字](img\image.png)

*注意：务必勾选底部的：Add Python 3.10 to PATH（如果不勾，你在命令行里输入 python 依然没反应）。*

验证
```
python --version
```

## 创建虚拟环境

ctrl shift p : python env

选择

*注意：记得关梯子*

## 安装

cpu的话安装cpu的torch：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install "numpy" "matplotlib" "opencv-python" "PyYAML"

## 👄

进入seam_pipeline或者seam_localization来探索吧🤣
