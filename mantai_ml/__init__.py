"""mantAI machine learning.

Two ways to feed the model, one shared record format:

    gameplay  : simulator games are logged decision by decision (recorder.py)
    video     : tiles are read from video frames and hand changes are turned
                into the same decision records (video/)

Records (JSON lines, see records.py) train a policy network (train.py) that
can then play as Player 1 in the simulator (ml_player.py, evaluate).

Run `python -m mantai_ml --help` from the project folder for the commands.
"""
