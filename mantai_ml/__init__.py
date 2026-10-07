"""mantAI machine learning.

Two ways to feed the model, one shared record format:

    gameplay  : simulator games are logged decision by decision (recorder.py)
    video     : tiles are read from video frames and hand changes are turned
                into the same decision records (video/)

Records (JSON lines, see records.py) train a policy network (train.py) that
can then play as Player 1 in the simulator (ml_player.py, evaluate).

The main player follows AlphaGo:
    knowledge.py : the study library in docs/ as features and a prior (the
                   "expert" knowledge AlphaGo took from human games)
    selfplay.py  : reinforcement learning by self-play, with a value network
    search.py    : before each discard, the network proposes candidates and
                   a search over guessed deals, scored by the value network
                   and rollouts, decides. The browser game uses it.

Run `python -m mantai_ml --help` from the project folder for the commands.
"""
