call conda env create -f environment.yaml
call conda activate rf_audio
call pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
:: Installing torch downgrades setuptools, reset it to original version
call pip install setuptools==80.8.0
call pip install opensoundscape==0.12.1
