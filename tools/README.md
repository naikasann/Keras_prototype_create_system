# tools

## 概要

学習する前に必要になる前処理のもろもろがあるフォルダー．

--------------

## labeltotext

画像のファイルパスとラベルのデータを並べたテキストを生成する

テキストは

``` .txt
[image_path] label(数値)
[image_path] label(数値)
[image_path] label(数値)
[image_path] label(数値)...
```

か

``` .txt
[image_path]
[image_path]
...
```

というテキストを生成する．
作成するためのフォルダー構成は

``` filedirectory
[指定ファイル]-[labelフォルダー]-[画像]
                              |-[画像]
                              ...
             -[labelフォルダー]-[画像]
                              |-[画像]
                              ...
             ...
```

か

``` filedirectory
[指定フォルダー]-[画像]
              |-[画像]
              ...
```

このような形のものをテキストにする．

実行はファイル内の

``` labeltotext.py
path = ""
```

を上の構成図の指定ファイルのディレクトリに変更

``` labeltotext.py
textfile = ""
```

を出力するテキストファイル名に変更し

``` commandshell
python labeltotext.py
```

で実行．

--------------

## resizeprog

画像を指定サイズにリサイズする

画像のファイル構成はlabeltotextと同様で

``` file directory
[指定ファイル]-[labelフォルダー]-[画像]
                              |-[画像]
                              ...
             -[labelフォルダー]-[画像]
                              |-[画像]
                              ...
             ...
```

または

``` file directory
[指定ファイル]-[画像]
            |-[画像]
            ...
```

このような形のものをすべてリサイズし画像を`上書き`する．

実行はファイル内の

``` resize.py
path = ""
```

を上の構成図の指定ファイルのディレクトリに変更

``` resize.py
size = ()
```

に指定サイズを入力する．入力は(width, height)のタプルを代入する，

``` command shell
python resizeprog.py
```

で実行．

--------------

## movietophoto

指定動画を切り出して画像にする

[Python, OpenCVで動画ファイルからフレームを切り出して保存 | note.nkmk.me](https://note.nkmk.me/python-opencv-video-to-still-image/)

を利用．自分用に書き換え予定.
