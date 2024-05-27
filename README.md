This is the code for our paper ```Privacy-Aware Randomized Quantization via Linear Programming```.

To obtain the results shown in Figure 3, run ```compare_err_uniform.py```. For instance:

```sh
python compare_err_uniform.py --c 1 --eps 1
```

To obtain the results shown in Table 2, run ```compare_err_gauss.py```.

To obtain the results shown in Figure 4(a), run ```vec_err.py```:

```sh
python vec_err.py --c 1 --type l1 --dim 10
```

To obtain the results shown in Figure 4(b), run ```vec_err.py```:

```sh
python vec_err.py --c 1 --type l2 --dim 100
```

To obtain the results shown in Figure 4(c), run ```dp_sgd_cancer.py```:

```sh
python dp_sgd_cancer.py --c 0.1 --eps 1
```

To obtain the results shown in Figure 4(d), run ```dp_sgd_mnist.py```:

```sh
python dp_sgd_cancer.py --c 0.01 --eps 1
```

The results of MVU mechanism can be recorded from [dp_compression](https://github.com/facebookresearch/dp_compression) project. 