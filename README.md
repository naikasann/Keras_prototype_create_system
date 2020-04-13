# KerasFramework
Kerasを使うときのフレームワーク

--------------
## 概要

Kerasを用いて学習するときに何度もコードを書くのがもったいないと感じたため，
ある程度なんでも対応できるように作成すれば手間かからないなあと感じたためフレームワークのような形で作成しようと考えた．

説明は各フォルダー内にREADMEを用意しておく

--------------
## ファイル構成
- /train --- 学習用のプログラムが入っているもの
- /test --- テスト用のプログラムが入る予定のもの(未作成)
- /tools --- 学習する際の前処理的な部分が入っている(きれいには書かないかも...)
- requirements.txt --- pythonのパッケージ管理ファイル

--------------
## 参考文献
1. [KerasのGeneratorを自作する - kumilog.net](https://www.kumilog.net/entry/keras-generator)
2. [Python OpenCVの基礎 resieで画像サイズを変えてみる - Pythonの学習の過程とか](https://peaceandhilightandpython.hatenablog.com/entry/2016/01/09/214333)
3. [Pythonでファイル名・ディレクトリ名の一覧をリストで取得 | note.nkmk.me](https://note.nkmk.me/python-listdir-isfile-isdir/)
4. [Python, OpenCVで動画ファイルからフレームを切り出して保存 | note.nkmk.me](https://note.nkmk.me/python-opencv-video-to-still-image/)
5. [pythonでyamlの扱い - Qiita](https://qiita.com/konoui/items/1d19aee73ff6e5135b73)