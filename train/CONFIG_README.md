# config.yamlのREADME

## 概要

config.yamlの各設定について

---

## Training

---

## runnning

実行時の環境設定．
(GPU使用するのかどうか)

---

## Resourcedata

データセットに関係する物の設定

- readdata --- データセットの読み取り方法を指定する．(text, onefolder, folder, bclearning)
- resourcepath --- データセットの呼び出し場所指定
- img_row --- 画像サイズ(width)
- img_col --- 画像サイズ(hight)
- classes --- カテゴリ．シーケンス表記で

### BCLearning

画像の学習手法の一つ。他のカテゴリの画像と画像をミックスして学習を行うというもの。
学習での境界区域周辺での認知能力がかなり低い場合に有効。
損失関数を `kullback_leibler_divergence` にすることを推奨する。
(学習効率が高い。)

[bclearningの論文リンク](https://arxiv.org/abs/1711.10284)

### classesの書き方

``` yaml
clsees :
    - category1 (ex - car)
    - category2 (ex - person)
    - category3 (ex - bike)
    - category4 (ex - dog)
    ...
```

と `-[カテゴリ名]` で書いていく

---

## Validation

バリデーションの有無の設定

- Usedata(重要) --- Validationを用いた学習を行うのか？(Falseの場合以下のValidationの設定をスキップする)
- readdata --- データセットの読み取り方法を指定する．(text, onefolder, folder)
- resourcepath --- データセットの呼び出し場所指定
- shuffle --- バリデーションのデータをシャッフルするのかどうか(いらない機能.お遊び)

readdata,resourcepathはResourcedataと同様．

---

## Trainsetting

trainを回すときの設定

- epoch --- エポック数指定
- batchsize --- バッチサイズの指定
- learnrate --- オプティマイザーの学習率の指定
- shuffle --- 学習データをシャッフルするのかどうかを判別する(バッチデータと学習する際の順番ともにシャッフルする)

---

## Modelsetting

モデルの設定

- model_loss --- モデルの損失関数指定
(現サポートはcategorical_crossentropy, mean_squared_error, binary_crossentropy, kullback_leibler_divergence)
- optimizers --- オプティマイザーの設定(現サポートはAdam, sgd, Adagrad, adadelta)
- network_architecture --- ネットワーク構造の設定(現在自由に作成中.ただ)
- retrain_model --- モデルをロードして再学習するのか．(転移学習やファインチューニングは未対応，おそらく別リポジトリに). Falseでした2つの設定をスキップする．
- model_path --- モデルの構造をロードする．(現在はyamlのモデルをロードするように)
- weight_path --- モデルの重みをロードする．(.h5ファイル)
- trainable --- モデルの重み更新を行うのかどうか．(Trueで学習．基本的にはこれで．)

---

## Trainingresult

学習結果についての設定

- model_name --- モデルを保存する際の名前を設定します．
(Alexnetを学習するときはAlexとかにすると識別が容易になると思います．)
- graph_write --- 実行後グラフを描画するのかどうかを設定します．

---

## Callback

学習時のコールバック設定(modelcheckpoint, tensorboard)

- monitor --- 監視対象の設定(loss, val_loss, acc, val_acc)
- verbose --- 詳細の表示設定(0: 表示なし, 1: 簡易表示, 2: 詳細表示)
- save_best_only --- 優良なモデルのみの保存なのか？
- save_weights_only --- モデルの重みのみ保存かどうか．(yamlで保存するようにしてあるのでTrueでもモデル保存される．)
- mode --- どれを監視するのか．(loss, val_lossはmin. acc, val_accはmaxがいい．)
- period --- callbackをどのタイミングで使用するか．(エポック指定)
- tensorboard --- tensorboardのパス指定
- tb_epoch --- tensorboardの更新エポックの指定(現在使用していない。すぐに導入できる。)

---

## Test

## TESTModel

テストのためのモデルの呼び込み設定

- path --- modelと重みのデータ，両方のデータが格納されているディレクトリを指定する．(リザルトの格納を容易にするため．)
- model_path --- テストに使用するモデルのアーキテクチャのディレクトリを指定する．
- weight_path --- テストで使用する学習の重みデータのディレクトリを指定する．
`[注意] モデル・重みのディレクトリは path + model_path, path + weight_pathで指定されている．テスト結果はpathに保存される．`

---

## testresourcedata

テストに使用するデータについて

- readdata --- trainと同様でデータの呼び込み方法
- resourcepath --- データの参照先
- img_row --- 画像サイズ(width)
- img_col --- 画像サイズ(hight)
- clasess --- カテゴリ(記述の仕方はTrainの方を参照)

---
