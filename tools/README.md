# tools

---------------
## 概要
Kerasにかける前に必要になる前処理のもろもろがあるフォルダー．

--------------
## ファイル構成
ファイル構成は以下の通り

--------------
- labeltotext --- 画像のファイルパスとラベルのデータを並べたテキストを生成するPython program

テキストは
```
[image_path] label(数値)
[image_path] label(数値)
[image_path] label(数値)
[image_path] label(数値)...
```
というテキストを生成する．
作成するためのフォルダー構成は

```ファイル構成
[指定ファイル]-[labelフォルダー]-[画像]
                              -[画像]
                              ...
             -[labelフォルダー]-[画像]
                              -[画像]
             ...
```
このような形のものをテキストにする．

--------------
- labeltotext --- 画像のファイルパスとラベルのデータを並べたテキストを生成するPython program

テキストは
```
[image_path] label(数値)
[image_path] label(数値)
[image_path] label(数値)
[image_path] label(数値)...
```
というテキストを生成する．
作成するためのフォルダーは別プログラムで説明