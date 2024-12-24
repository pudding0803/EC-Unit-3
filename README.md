# Map Generator Through Evolutionary Algorithm

> A project for Unit 3 of the Evolutionary Computation course

## Methods Overview

### Initialization

- Generate initial maps using Perlin noise to simulate elevation, categorizing terrain into mountains, rivers, and grass based on elevation values.
- Apply cellular automata multiple times to smooth rough transitions and create more natural patterns.

### Terrain Refinement

- Randomly swap small portions of grass with rocks and rivers with river stones. This exchange is bidirectional, meaning rocks may turn back into grass, and river stones can revert to rivers.
- The refinement ensures consistency since the swapped terrains share similar contexts.

## Genetic Operations

### Selection

- Tournament selection is used to choose parent maps for reproduction, giving individuals with higher fitness a better chance of selection.

### Crossover

- A random rectangular area is selected, and terrain values between two parents are swapped.

### Mutation

- Individual terrain cells are altered with a small probability, based on predefined mutation weights.
- After mutation, cellular automata are reapplied to smooth transitions, followed by additional random terrain swaps for variety.

### Deterministic Crowding

- Compare offspring with their respective parents based on similarity and fitness. This helps preserve well-adapted features while maintaining diversity.

## Fitness Evaluation

### Goals

- Create basin and delta landscapes within the map.
- Simulate realistic river widths: rivers become narrower at higher altitudes and wider at lower altitudes.

### Water Body Classification

1. Start by marking all edge water tiles as ocean.
2. Perform DFS on all ocean tiles:
   - If 80% of the surrounding tiles are water, classify the tile as ocean.
   - Otherwise, temporarily mark it as an ocean boundary.
3. Perform DFS on all ocean boundary tiles:
   - If at least 20% of the surrounding tiles are ocean, reclassify the tile as ocean.
4. Mark unvisited water tiles with at least 80% water surroundings as lake tiles.
5. Perform DFS on all lake tiles:
   - If 80% of the surrounding tiles are water, classify the tile as a lake.
   - Otherwise, temporarily mark it as a lake boundary.
6. Perform DFS on all lake boundary tiles:
   - If at least 20% of the surrounding tiles are lake, reclassify the tile as lake.
7. Any remaining water tiles are classified as rivers.

### River Segmentation

For each river tile, classify it into one of three segments based on the surrounding terrain:

- If mountain tiles outnumber grass tiles by more than $\delta$, classify it as upstream.
- If grass tiles outnumber mountain tiles by more than $\delta$, classify it as downstream.
- Otherwise, classify it as midstream.

### Basin Evaluation

For each grass tile, if enough surrounding tiles are mountains, increase the fitness score.

### Delta Evaluation

For each riverstone tile located near a river mouth (surrounded by both downstream river tiles and ocean tiles), increase the fitness score.

### River Width Evaluation

For each river tile:

- Penalize upstream tiles that are too wide (if the surrounding river tiles exceed a threshold).
- Penalize downstream tiles that are too narrow (if the surrounding river tiles fall below a threshold).

### Isolation Penalty

For any tile surrounded by fewer than two tiles of the same type, reduce the fitness score. If the tile has no directly adjacent neighbors of the same type, apply a larger penalty.

### Map Proportion Evaluation

- Calculate the proportion of each terrain type across the map and compare it to ideal ratios:
  - Mountains: 30%
  - Grass (including rock): 30%
  - Rivers (including river stones): 40%
- Apply a penalty based on the deviation from these ideal ratios, with a higher weight since this evaluation impacts the map's overall balance.

## Discussion

- Ten generated maps are displayed in the `output` folder.
- Most maps feature basin-like landscapes, though it’s unclear whether this results from the fitness function design or coincidence.
- River connectivity remains an issue, and no solution beyond manual adjustments has been implemented. Consequently, the evaluation of deltas and river widths is limited.
- Using CNNs in the future could provide a better approach for extracting and evaluating map features.
