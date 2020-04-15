# tools

## 概要
Kerasにかける前に必要になる前処理のもろもろがあるフォルダー．

--------------
## labeltotext
画像のファイルパスとラベルのデータを並べたテキストを生成する

テキストは
```
[image_path] label(数値)
[image_path] label(数値)
[image_path] label(数値)
[image_path] label(数値)...
```
というテキストを生成する．
作成するためのフォルダー構成は

```
[指定ファイル]-[labelフォルダー]-[画像]
                              |-[画像]
                              ...
             -[labelフォルダー]-[画像]
                              |-[画像]
                              ...
             ...
```
このような形のものをテキストにする．

実行はファイル内の
```
path = ""
```
を上の構成図の指定ファイルのディレクトリに変更
```
textfile = ""
```
を出力するテキストファイル名に変更し
```
python labeltotext.py
```
で実行．

--------------
## resizeprog
画像を指定サイズにリサイズする

画像のファイル構成はlabeltotextと同様で
```
[指定ファイル]-[labelフォルダー]-[画像]
                              |-[画像]
                              ...
             -[labelフォルダー]-[画像]
                              |-[画像]
                              ...
             ...
```
このような形のものをすべてリサイズ(`上書き`)にする．

実行はファイル内の
```
path = ""
```
を上の構成図の指定ファイルのディレクトリに変更
```
size = ()
```
に指定サイズを入力する．入力は(width, height)のタプルを代入する，
```
python resizeprog.py
```
で実行．

--------------
## movietophoto
指定動画を切り出して画像にする

[Python, OpenCVで動画ファイルからフレームを切り出して保存 | note.nkmk.me](https://note.nkmk.me/python-opencv-video-to-still-image/)

を利用．自分用に書き換え予定．