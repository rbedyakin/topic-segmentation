# Topic Segmentation

## Description
This repository hosts the code for Topic Segmentation.

## Introduction to Topic Segmentation
Topic Segmentation is the task of splitting text into meaningful segments that correspond to distinct topics.

### Example

#### Input:
Beer checkers (also known as shotglass checkers, shot glass checkers or beercheckers) is a two player drinking game. A variant of normal checkers, it is played on a standard checkerboard (or chess board), using cups of beer in place of the regular checker pieces (or draughts). The game is popular with students at universities in the United States. The board is set up as for a normal game of checkers, with 12 cups to a side, each cup half-full of beer. Either the cups should be visually distinct to distinguish the players' pieces, or a light and a dark beer should be used for each player. The game is played by the standard rule for English draughts. When a piece reaches the king's row, it is designated as a king by filling the cup full. When a piece is jumped by an opposing player, the owner of the piece must drink the beer from the jumped piece. The game is won by a player when the other play cannot move their piece, either because all the remaining pieces are blocked, or because the player has no remaining pieces. The losing player must drink all remaining beers on the board. A pivotal scene in Graham Greene's 1958 novel "Our Man in Havana" sees the protagonist playing a game of draughts using miniature bottles of scotch and bourbon. In the film Love Is News (1937), Tyrone Power plays beer checkers with his fellow newspaper reporters.</div>
#### Output:
<span style="color:blue">&nbsp;&nbsp;&nbsp;&nbsp;Beer checkers (also known as shotglass checkers, shot glass checkers or beercheckers) is a two player drinking game. A variant of normal checkers, it is played on a standard checkerboard (or chess board), using cups of beer in place of the regular checker pieces (or draughts). The game is popular with students at universities in the United States.</span><br><span style="color:green">&nbsp;&nbsp;&nbsp;&nbsp;The board is set up as for a normal game of checkers, with 12 cups to a side, each cup half-full of beer. Either the cups should be visually distinct to distinguish the players' pieces, or a light and a dark beer should be used for each player. The game is played by the standard rule for English draughts. When a piece reaches the king's row, it is designated as a king by filling the cup full. When a piece is jumped by an opposing player, the owner of the piece must drink the beer from the jumped piece. The game is won by a player when the other play cannot move their piece, either because all the remaining pieces are blocked, or because the player has no remaining pieces. The losing player must drink all remaining beers on the board.</span><br><span style="color:red">&nbsp;&nbsp;&nbsp;&nbsp;A pivotal scene in Graham Greene's 1958 novel "Our Man in Havana" sees the protagonist playing a game of draughts using miniature bottles of scotch and bourbon. In the film Love Is News (1937), Tyrone Power plays beer checkers with his fellow newspaper reporters.</span></div>

## Installation
Install all required python dependencies:
```
pip install -r requirements.txt
```

## How to use

### LLM (Gemma3, LLama3, ...)

#### Preprocessing
```
python ./src/preprocess.py 
```

#### Finetune
```
python ./src/llm_lora.py 
```

## Data
### WikiSection
GitHub dataset page: https://github.com/sebastianarnold/WikiSection <br>
Please refer to [SECTOR: A Neural Model for Coherent Topic Segmentation and Classification](https://arxiv.org/abs/1902.04793) for details about the dataset construction.

### Wiki-727k
HuggingFace dataset page: https://huggingface.co/datasets/TankNee/wiki-727k <br>
Please refer to [Text Segmentation as a Supervised Learning Task](https://arxiv.org/abs/1803.09337) for details about the dataset construction.

### YTSeg
HuggingFace dataset page: https://huggingface.co/datasets/retkowski/ytseg <br>
Please refer to [From Text Segmentation to Smart Chaptering: A Novel Benchmark for Structuring Video Transcriptions](https://arxiv.org/abs/2402.17633) for details about the dataset construction.

## Models

- [x] Gemma3 <br>
