python -m pip install virtualenv
python -m virtualenv venv
call venv\scripts\activate.bat
curl --location --remote-header-name --remote-name https://github.com/RichardErkhov/FastFaceSwap/releases/download/model/inswapper_128.onnx
curl --location --remote-header-name --remote-name https://github.com/RichardErkhov/FastFaceSwap/releases/download/model/GFPGANv1.4.pth
curl --location --remote-header-name --remote-name https://github.com/RichardErkhov/FastFaceSwap/releases/download/model/256_v2_big_generator_pretrain_stage1_38499.h5
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install tensorflow-gpu==2.10.1
pip install protobuf==3.20.2

pip uninstall opencv-python opencv-headless-python opencv-contrib-python -q -y
pip install opencv-python
