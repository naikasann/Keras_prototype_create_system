# Keras_prototype_create_system

Kerasのクラス分類を様々な方法で行えるように作成したもの。

--------------

## 概要

Kerasを用いて学習するときに何度もコードを書くのがもったいないと感じたため，
ある程度なんでも対応できるように作成すれば手間かからないなあと感じたためフレームワーク（フレームワークのフレームワークっておかしいね）
のような形で作成しようと考えた．

説明は各フォルダー内にREADMEを用意しておく

またKerasなどの深層学習フレームワークのプログラムの知識が深くない者でも深層学習を行うことができるように設計した。
（モデルの作成や学習方法などのプログラムの作り方がわからない人向け。）

結構変数の分け方とかは適当なのは許してもらえると助かります。。。

--------------

## ファイル構成

ファイル構成は以下の通り

- /train --- 学習用のプログラムが入っているもの
- /tools --- 学習する際の前処理的な部分が入っている(きれいには書かないかも...)
- /model_sample --- 学習モデルのサンプル置き場

--------------
## camera.py

ウェブカメラを利用して推論を行うというもの
``` python camera.py ```
で実行可能

--------------
## バージョン
tensorflow 2.x系（開発は2.2 & 2.3で開発。）
その他は適当。

--------------

## 今後の予定

このプログラムは現在の段階で固定し、なにか再利用できそうな場合のみ再利用していく。
またこの方法は知識がない人でも学習を行うことができるという点で評価できたので、
今後はその点を特化できるように、ウェブフレームワークと混合した、深層学習を軽くできる
プラットフォームの作成ができたらいいかな。。

--------------


## 参考文献

1. [KerasのGeneratorを自作する - kumilog.net](https://www.kumilog.net/entry/keras-generator)
2. [Python OpenCVの基礎 resieで画像サイズを変えてみる - Pythonの学習の過程とか](https://peaceandhilightandpython.hatenablog.com/entry/2016/01/09/214333)
3. [Pythonでファイル名・ディレクトリ名の一覧をリストで取得 | note.nkmk.me](https://note.nkmk.me/python-listdir-isfile-isdir/)
4. [Python, OpenCVで動画ファイルからフレームを切り出して保存 | note.nkmk.me](https://note.nkmk.me/python-opencv-video-to-still-image/)
5. [pythonでyamlの扱い - Qiita](https://qiita.com/konoui/items/1d19aee73ff6e5135b73)
6. [bclearningの論文リンク](https://arxiv.org/abs/1711.10284)
7. [【Python】 二つのリストの対応関係を保ったままシャッフルする - 旅行好きなソフトエンジニアの備忘録](https://mail.google.com/mail/u/2/?tab=wm&ogbl#inbox/FMfcgxwHNWJQmWdZwjDBJHsLcZWGHfTq)
8. [kerasのto_categoricalを使ってみる | 分析ノート](https://analytics-note.xyz/machine-learning/keras-to-categorical/)
9. [速くて軽くて精度の良い、MobileNetのポイントをまとめてみた - Qiita](https://qiita.com/simonritchie/items/f6d6196b1b0c41ca163c)
10. [Pythonでカメラを制御する【研究用】 - Qiita](https://qiita.com/opto-line/items/7ade854c26a50a485159)
11. [Python+OpenCVでカメラキャプチャ - Qiita](https://qiita.com/wkentaro/items/3d3bee56445894da879e)
12. [pandasでcsv/tsvファイル読み込み（read_csv, read_table） | note.nkmk.me](https://note.nkmk.me/python-pandas-read-csv-tsv/)
13. [Grad CAM implementation with Tensorflow 2](https://gist.github.com/RaphaelMeudec/e9a805fa82880876f8d89766f0690b54)
14. [Interpretability of Deep Learning Models with Tensorflow 2.0](https://www.sicara.ai/blog/2019-08-28-interpretability-deep-learning-tensorflow)
15. [Grad-CAMでヒートマップを表示 - Qiita](https://qiita.com/yakisobamilk/items/8f094590e5f45a24b59c)
16. [Social Network for Programmers and Developers](https://morioh.com/p/64064daff26c)
17. [Keras - Keras の ImageDataGenerator を使って学習画像を増やす - Pynote](https://www.pynote.info/entry/keras-image-data-generator)
18. [ディープラーニング　脱超初心者向け基礎知識 - Qiita](https://qiita.com/gal1996/items/00ed3589e13448496b4c)
19. [Tutorial on Keras flow_from_dataframe | by Vijayabhaskar J | Medium](https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c)
20. [pandas.DataFrame, SeriesとPython標準のリストを相互に変換 | note.nkmk.me](https://note.nkmk.me/python-pandas-list/)
21. [Image data preprocessing](https://keras.io/api/preprocessing/image/)
22. [[TF]KerasでModelとParameterをLoad/Saveする方法 - Qiita](https://qiita.com/supersaiakujin/items/b9c9da9497c2163d5a74)
23. [KerasでAlexNetを構築しCifar-10を学習させてみた - Qiita](https://qiita.com/_uran_0831/items/ea2bfc8f7ba2fc858de3)
24. [keras.preprocessing.image使い方メモ - Qiita](https://qiita.com/tom_eng_ltd/items/aed56e8c42657e22bc4c)
25. [LearningRateScheduler](https://keras.io/api/callbacks/learning_rate_scheduler/)
26. [KerasのLearningRateSchedulerを使って学習率を変化させる | Shikoan's ML Blog](https://blog.shikoan.com/keras-learning-rate-decay/)
27. [「Kerasのto_categoricalの挙動ってちょっと変わってるよね」という話 - Ahogrammer](https://hironsan.hatenablog.com/entry/keras-to-categorical)

and more...

---

このリポジトリについて
大学学部~大学院1年次に書いたものでだいぶ機能も使い勝手も悪いです
