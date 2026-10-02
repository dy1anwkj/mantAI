"""Video input: read Mahjong tiles from gameplay footage.

    frames.py      read frames from a video file at a chosen rate
    detect.py      find tile-shaped boxes in a region of the frame
    classifier.py  a small CNN that names each tile crop (46 labels)
    crops.py       cut tile images out of a video so you can label them
    reader.py      turn a video into decision records (same format as gameplay)
    synth.py       synthetic tiles and videos for demos and tests
"""
