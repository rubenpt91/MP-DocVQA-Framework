python3 eval.py -m BigBird -d DUDE -bs 1 --seed 42
python3 train.py -m hi-vt5 -d DUDE -bs 1 --seed 42
python3 train.py -m LayoutLMv3 -d DUDE -bs 1 --seed 42




python3 train.py -m BigBird -d DUDE-sample -bs 2 --seed 42 --no-eval-start
nohup python3 train.py -m BigBird -d DUDE -bs 2 --seed 42 &

#logits mode is not able to run [loss undefined]