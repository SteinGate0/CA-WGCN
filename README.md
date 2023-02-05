# CA-WGCN

This paper will be published in the Journal of Applied Intelligence in 2021.

*Weighted graph convolution over dependency trees for nontaxonomic relation extraction on public opinion information*[(link)](https://doi.org/10.1007/s10489-021-02596-9) 

Thanks to iFlytek processing platform for its help in data set processing. The platform link is as follows:[(link)](https://www.xfyun.cn/) 

## Prepare vocab embedding

To generate vocab embedding and word list(include entity, pos... label), run:

```
python train.py  prepare_vocab.py
```
Generate vocab.pkl and  embedding.npy file

## Training 

To train a graph convolutional neural network (CA-WGCN) model, run:

```
python train.py
```
Model checkpoints and logs will be saved to `./saved_models/00`.

Next training  automatic
Model checkpoints and logs will be saved to `./saved_models/01`.

For details on the use of other parameters, such as the pruning distance k, please refer to `train.py`.

 ## Evaluation

To run evaluation on the test set, run:

```
python eval.py saved_models/00 --dataset test
```
This will use the `best_model.pt` file by default. Use `--model checkpoint_epoch_10.pt` to specify a model checkpoint file.

## Retrain

Reload a pretrained model and finetune it, run:

```
python train.py --load --model_file saved_models/01/best_model.pt --optim sgd --lr 0.001
```

## References for Code Base

C-GCN: [Pytorch repo](https://github.com/qipeng/gcn-over-pruned-trees); this is the origin of the My Code.

The pre-training Chinese word vector used in the experiment comes from the baidu Encyclopedia pre-training word vector[(link)](https://github.com/Embedding/Chinese-Word-Vectors)  , placed under the file path Dataset/baidu of this project.


## Citation

Please cite our work if you find it useful:
```
Wang, G., Liu, S. & Wei, F. Weighted graph convolution over dependency trees for nontaxonomic relation extraction on public opinion information. Appl Intell 52, 3403–3417 (2022).
```
