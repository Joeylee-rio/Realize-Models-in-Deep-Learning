# Realize-Models-in-Deep-Learning
#### I hope this will be a tutorial for freshs to Deep-Learning
##### Main References:
1: nlp-tutorial repo authored by graycode(Tae-Hwan Jung et.al.) : https://github.com/graykode/nlp-tutorial
2: Prof.Hungyi Lee@NTU(National Taiwan Univ.) and his tutorials : http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html
3: Deep Learning Tutorials by Mu Li@Amazon : https://space.bilibili.com/1567748478/channel/seriesdetail?sid=358497
## synopsis
### 1: CNN
### 2: RNN
### 3: Seq2Seq(based on RNN)
#### 3.1: seq2seq(without attention mechanism)
#### 3.2: seq2seq(with attention decoder)
### 4: Transformer
### 5: Pretrained Language Models
#### 5.1: BERT
#### 5.2: T5(text-to-text transfer tranformer)
### 6: GAN
### 7: Meta-Learning(MAML)
### 8: left for complement

### python-package-envs:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main    defaults
_openmp_mutex             4.5                       1_gnu    defaults
aiohttp                   3.8.1            py37h7f8727e_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
aiosignal                 1.2.0              pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
async-timeout             4.0.1              pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
asynctest                 0.13.0                     py_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
attrs                     21.2.0             pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
blas                      1.0                         mkl    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
boto3                     1.18.21            pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
botocore                  1.21.41            pyhd3eb1b0_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
brotlipy                  0.7.0           py37h27cfd23_1003    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
bzip2                     1.0.8                h7b6447c_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
c-ares                    1.17.1               h27cfd23_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
ca-certificates           2021.10.26           h06a4308_2    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
cachetools                4.2.2              pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
certifi                   2021.10.8        py37h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
cffi                      1.14.6           py37h400218f_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
charset-normalizer        2.0.4              pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
cryptography              35.0.0           py37hd23ed53_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
cudatoolkit               10.2.89              hfd86e86_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
cython                    0.29.23          py37h2531618_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
ffmpeg                    4.3                  hf484d3e_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
freetype                  2.11.0               h70c0345_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
frozenlist                1.2.0            py37h7f8727e_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
gensim                    4.0.1            py37h2531618_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
giflib                    5.2.1                h7b6447c_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
gmp                       6.2.1                h2531618_2    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
gnutls                    3.6.15               he1e5248_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
google-api-core           1.25.1             pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
google-auth               1.33.0             pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
google-cloud-core         1.7.1              pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
google-cloud-storage      1.41.0             pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
google-crc32c             1.1.2            py37h27cfd23_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
google-resumable-media    1.3.1              pyhd3eb1b0_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
googleapis-common-protos  1.53.0           py37h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
grpcio                    1.42.0           py37hce63b2e_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
idna                      3.3                pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
intel-openmp              2021.4.0          h06a4308_3561    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
jmespath                  0.10.0             pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
jpeg                      9d                   h7f8727e_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
lame                      3.100                h7b6447c_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
lcms2                     2.12                 h3be6417_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
ld_impl_linux-64          2.35.1               h7274673_9    defaults
libcrc32c                 1.1.1                he6710b0_2    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libffi                    3.3                  he6710b0_2    defaults
libgcc-ng                 9.3.0               h5101ec6_17    defaults
libgfortran-ng            7.5.0               ha8ba4b0_17    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libgfortran4              7.5.0               ha8ba4b0_17    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libgomp                   9.3.0               h5101ec6_17    defaults
libiconv                  1.15                 h63c8f33_5    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libidn2                   2.3.2                h7f8727e_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libpng                    1.6.37               hbc83047_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libprotobuf               3.17.2               h4ff587b_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libstdcxx-ng              9.3.0               hd4cf53a_17    defaults
libtasn1                  4.16.0               h27cfd23_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libtiff                   4.2.0                h85742a9_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libunistring              0.9.10               h27cfd23_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libuv                     1.40.0               h7b6447c_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libwebp                   1.2.0                h89dd481_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libwebp-base              1.2.0                h27cfd23_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
lz4-c                     1.9.3                h295c915_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
mkl                       2021.4.0           h06a4308_640    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
mkl-service               2.4.0            py37h7f8727e_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
mkl_fft                   1.3.1            py37hd3c417c_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
mkl_random                1.2.2            py37h51133e4_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
multidict                 5.1.0            py37h27cfd23_2    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
ncurses                   6.2                  he6710b0_1    defaults
nettle                    3.7.3                hbbd107a_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
numpy                     1.21.2           py37h20f2e39_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
numpy-base                1.21.2           py37h79a1101_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
olefile                   0.46                     py37_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
openh264                  2.1.0                hd408876_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
openssl                   1.1.1l               h7f8727e_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pillow                    8.4.0            py37h5aabda8_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pip                       21.0.1           py37h06a4308_0    defaults
protobuf                  3.17.2           py37h295c915_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pyasn1                    0.4.8              pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pyasn1-modules            0.2.8                      py_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pycparser                 2.21               pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pyopenssl                 21.0.0             pyhd3eb1b0_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pysocks                   1.7.1                    py37_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
python                    3.7.11               h12debd9_0    defaults
python-dateutil           2.8.2              pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pytorch                   1.10.0          py3.7_cuda10.2_cudnn7.6.5_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
pytorch-mutex             1.0                        cuda    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
pytz                      2021.3             pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
readline                  8.1                  h27cfd23_0    defaults
requests                  2.26.0             pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
rsa                       4.7.2              pyhd3eb1b0_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
s3transfer                0.5.0              pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
scipy                     1.7.1            py37h292c36d_2    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
setuptools                58.0.4           py37h06a4308_0    defaults
six                       1.16.0             pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
smart_open                5.1.0              pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
sqlite                    3.36.0               hc218d9a_0    defaults
tk                        8.6.10               hbc83047_0    defaults
torchaudio                0.10.0               py37_cu102    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
torchvision               0.11.1               py37_cu102    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
typing-extensions         3.10.0.2             hd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
typing_extensions         3.10.0.2           pyh06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
urllib3                   1.26.7             pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
wheel                     0.37.0             pyhd3eb1b0_1    defaults
xz                        5.2.5                h7b6447c_0    defaults
yarl                      1.5.1            py37h7b6447c_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
zlib                      1.2.11               h7b6447c_3    defaults
zstd                      1.4.9                haebb681_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
