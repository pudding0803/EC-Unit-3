import os
import random
from enum import Enum

import numpy as np
from PIL import Image
from noise import pnoise2
from tqdm import tqdm


class Terrain(Enum):
    MOUNTAIN = 0
    RIVER = 1
    GRASS = 2
    ROCK = 3
    RIVER_STONE = 4


class WaterType(Enum):
    UNKNOWN = 'unknown'
    OCEAN = 'ocean'
    LAKE = 'lake'
    RIVER = 'river'
    BOUNDARY = 'boundary'


class RiverSegment(Enum):
    MIDSTREAM = 0
    UPSTREAM = 1
    DOWNSTREAM = 2


image_paths = {
    Terrain.MOUNTAIN.value: './tiles/mountain.png',
    Terrain.RIVER.value: './tiles/river.png',
    Terrain.GRASS.value: './tiles/grass.png',
    Terrain.ROCK.value: './tiles/rock.png',
    Terrain.RIVER_STONE.value: './tiles/riverstone.png'
}

rows, columns = 50, 50
population = 100
generations = 10
mutation_rate = 0.2
mutation_weights = [3, 4, 3]
scale = 0.1
output_dir = 'output'


def replace_terrain(digit_map):
    for y in range(rows):
        for x in range(columns):
            if digit_map[y][x] == Terrain.GRASS.value and random.random() < 0.03:
                digit_map[y][x] = Terrain.ROCK.value
            elif digit_map[y][x] == Terrain.ROCK.value and random.random() < 0.3:
                digit_map[y][x] = Terrain.GRASS.value
            elif digit_map[y][x] == Terrain.RIVER.value and random.random() < 0.05:
                digit_map[y][x] = Terrain.RIVER_STONE.value
            elif digit_map[y][x] == Terrain.RIVER_STONE.value and random.random() < 0.3:
                digit_map[y][x] = Terrain.RIVER.value
    return digit_map


def apply_cellular_automata(digit_map, iterations=3):
    digit_map = np.array(digit_map)
    for _ in range(iterations):
        new_map = digit_map.copy()
        for y in range(rows):
            for x in range(columns):
                neighbors = [
                    digit_map[ny, nx]
                    for dy in [-1, 0, 1] for dx in [-1, 0, 1]
                    if not (dx == 0 and dy == 0)
                       and 0 <= (ny := y + dy) < rows
                       and 0 <= (nx := x + dx) < columns
                ]
                most_common = max(set(neighbors), key=neighbors.count)
                if neighbors.count(most_common) >= 5:
                    new_map[y, x] = most_common
        digit_map = new_map
    return digit_map.tolist()


def initialize_digit_maps():
    digit_maps = []
    for _ in range(population):
        noise_map = [
            [
                Terrain.MOUNTAIN.value if pnoise2(x * scale, y * scale, base=random.randint(0, 100)) > 0.1 else
                Terrain.RIVER.value if pnoise2(x * scale, y * scale, base=random.randint(0, 100)) < -0.05 else
                Terrain.GRASS.value
                for x in range(columns)
            ]
            for y in range(rows)
        ]
        noise_map = apply_cellular_automata(noise_map)
        digit_maps.append(replace_terrain(noise_map))
    return digit_maps


def classify_water_bodies(digit_map):
    water_data = {
        (y, x): {'type': WaterType.UNKNOWN, 'water_ratio': 0.0, 'visited': False}
        for y in range(rows) for x in range(columns)
        if digit_map[y][x] in [Terrain.RIVER.value, Terrain.RIVER_STONE.value]
    }

    # Calculate the water ratio around each water cell
    for (y, x) in water_data:
        neighbors = [
            digit_map[ny][nx]
            for dy in range(-2, 3)
            for dx in range(-2, 3)
            if 0 <= (ny := y + dy) < rows and 0 <= (nx := x + dx) < columns
        ]
        water_data[(y, x)]['water_ratio'] = sum(
            1 for n in neighbors if n in [Terrain.RIVER.value, Terrain.RIVER_STONE.value]) / len(neighbors)

    def traverse_water(start_list, target_type, ratio_threshold, boundary_list):
        current_list = start_list[:]
        while current_list:
            cy, cx = current_list.pop()
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if (ny, nx) in water_data and not water_data[(ny, nx)]['visited']:
                    water_data[(ny, nx)]['visited'] = True
                    if water_data[(ny, nx)]['water_ratio'] >= ratio_threshold:
                        water_data[(ny, nx)]['type'] = target_type
                        current_list.append((ny, nx))
                    else:
                        water_data[(ny, nx)]['type'] = WaterType.BOUNDARY
                        boundary_list.append((ny, nx))

    def refine_boundary(boundary_list, target_type, neighbor_ratio):
        while boundary_list:
            cy, cx = boundary_list.pop()
            neighbors = [
                (ny, nx)
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if (ny := cy + dy, nx := cx + dx) in water_data
            ]
            type_count = sum(1 for (ny, nx) in neighbors if water_data[(ny, nx)]['type'] == target_type)
            if type_count / len(neighbors) >= neighbor_ratio:
                water_data[(cy, cx)]['type'] = target_type

    # Mark "ocean"
    ocean_start = []
    boundary_list = []
    for (y, x) in water_data:
        if y == 0 or y == rows - 1 or x == 0 or x == columns - 1:
            water_data[(y, x)]['type'] = WaterType.OCEAN
            water_data[(y, x)]['visited'] = True
            ocean_start.append((y, x))

    traverse_water(ocean_start, target_type=WaterType.OCEAN, ratio_threshold=0.8, boundary_list=boundary_list)
    refine_boundary(boundary_list, target_type=WaterType.OCEAN, neighbor_ratio=0.2)

    for (y, x) in water_data:
        if water_data[(y, x)]['type'] == WaterType.BOUNDARY:
            water_data[(y, x)]['type'] = WaterType.OCEAN

    # Mark "lake"
    lake_start = [
        (y, x)
        for (y, x) in water_data
        if not water_data[(y, x)]['visited'] and water_data[(y, x)]['water_ratio'] >= 0.8
    ]
    traverse_water(lake_start, target_type=WaterType.LAKE, ratio_threshold=0.8, boundary_list=boundary_list)
    refine_boundary(boundary_list, target_type=WaterType.LAKE, neighbor_ratio=0.2)

    # Mark "river"
    for (y, x) in water_data:
        if water_data[(y, x)]['type'] == WaterType.UNKNOWN:
            water_data[(y, x)]['type'] = WaterType.RIVER

    return water_data


def calc_fitness(digit_map):
    fitness = 0

    water_types = classify_water_bodies(digit_map)
    river_segments = np.zeros((rows, columns), dtype=int)
    terrain_count = {terrain: 0 for terrain in Terrain}

    for y in range(rows):
        for x in range(columns):
            terrain_count[Terrain(digit_map[y][x])] += 1

            neighbors_5x5 = [
                (ny, nx)
                for dy in range(-2, 3)
                for dx in range(-2, 3)
                if 0 <= (ny := y + dy) < rows and 0 <= (nx := x + dx) < columns
            ]
            neighbors_3x3 = [
                (ny, nx)
                for dy in range(-1, 2)
                for dx in range(-1, 2)
                if 0 <= (ny := y + dy) < rows and 0 <= (nx := x + dx) < columns
            ]

            # Basin scoring
            if digit_map[y][x] in [Terrain.GRASS.value, Terrain.ROCK.value]:
                mountain_count = neighbors_5x5.count(Terrain.MOUNTAIN.value)
                grass_count = neighbors_5x5.count(Terrain.GRASS.value) + neighbors_5x5.count(Terrain.ROCK.value)
                if mountain_count > grass_count:
                    fitness += 1

            # Classify rivers into upstream, midstream, and downstream
            if digit_map[y][x] == Terrain.RIVER.value:
                mountain_count = neighbors_5x5.count(Terrain.MOUNTAIN.value)
                grass_count = neighbors_5x5.count(Terrain.GRASS.value) + neighbors_5x5.count(Terrain.ROCK.value)
                if mountain_count - grass_count > 3:
                    river_segments[y][x] = RiverSegment.UPSTREAM.value
                elif grass_count - mountain_count > 3:
                    river_segments[y][x] = RiverSegment.DOWNSTREAM.value

            # Delta scoring
            if digit_map[y][x] == Terrain.RIVER_STONE.value:
                river_rock_count = sum(1 for ny, nx in neighbors_5x5 if digit_map[ny][nx] == Terrain.RIVER_STONE.value)
                ocean_count = sum(
                    1 for ny, nx in neighbors_5x5 if (ny, nx) in water_types and water_types[ny, nx] == WaterType.OCEAN
                )
                downstream_count = sum(
                    1 for ny, nx in neighbors_5x5 if
                    (ny, nx) in water_types and river_segments[ny, nx] == RiverSegment.DOWNSTREAM.value
                )
                if river_rock_count >= 3 and ocean_count > 3 and downstream_count > 3:
                    fitness += 1

            # River width scoring
            if digit_map[y][x] == Terrain.RIVER.value:
                river_types = neighbors_3x3.count(Terrain.RIVER.value) + neighbors_3x3.count(Terrain.RIVER_STONE.value)
                if river_segments[y][x] == RiverSegment.UPSTREAM.value and river_types >= 2:
                    fitness -= 3
                elif river_segments[y][x] == RiverSegment.DOWNSTREAM.value and river_types <= 2:
                    fitness -= 3

            # Isolation penalty
            same_type_count = neighbors_3x3.count(digit_map[y][x])

            if same_type_count < 2:
                fitness -= 2
                direct_neighbors = [
                    digit_map[ny][nx]
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    if 0 <= (ny := y + dy) < rows and 0 <= (nx := x + dx) < columns
                ]
                if direct_neighbors.count(digit_map[y][x]) == 0:
                    fitness -= 3

    # Map proportion scoring
    total_cells = rows * columns
    mountain_ratio = terrain_count[Terrain.MOUNTAIN] / total_cells
    grass_ratio = (terrain_count[Terrain.GRASS] + terrain_count[Terrain.ROCK]) / total_cells
    river_ratio = (terrain_count[Terrain.RIVER] + terrain_count[Terrain.RIVER_STONE]) / total_cells

    ideal_ratios = [0.3, 0.3, 0.4]
    actual_ratios = [mountain_ratio, grass_ratio, river_ratio]
    ratio_score = -sum(abs(ideal - actual) for ideal, actual in zip(ideal_ratios, actual_ratios))
    fitness += ratio_score * 10

    return fitness


def tournament_selection(digit_maps, fitness_scores):
    tournament = random.sample(list(zip(digit_maps, fitness_scores)), 2)
    parent1 = max(tournament, key=lambda x: x[1])[0]
    tournament = random.sample(list(zip(digit_maps, fitness_scores)), 2)
    parent2 = max(tournament, key=lambda x: x[1])[0]
    return parent1, parent2


def crossover(parent1, parent2):
    y_start = random.randint(0, rows - 1)
    y_end = random.randint(0, rows - 1)
    x_start = random.randint(0, columns - 1)
    x_end = random.randint(0, columns - 1)

    child1 = [row[:] for row in parent1]
    child2 = [row[:] for row in parent2]

    for y in range(y_start, y_end + 1):
        for x in range(x_start, x_end + 1):
            child1[y][x] = parent2[y][x]
            child2[y][x] = parent1[y][x]

    return child1, child2


def mutation(digit_map):
    for row in range(rows):
        for col in range(columns):
            if random.random() < mutation_rate:
                digit_map[row][col] = random.choices(
                    [Terrain.MOUNTAIN.value, Terrain.RIVER.value, Terrain.GRASS.value],
                    weights=mutation_weights
                )[0]


def save_maps(digit_maps, top_indices):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for rank, idx in enumerate(top_indices):
        map_data = digit_maps[idx]
        image = Image.new('RGB', (columns * 64, rows * 64))
        for row_idx, row in enumerate(map_data):
            for col_idx, value in enumerate(row):
                tile = Image.open(image_paths[value])
                image.paste(tile, (col_idx * 64, row_idx * 64))
        image.save(os.path.join(output_dir, f'map_rank_{rank + 1}.png'))


def distance(map1, map2):
    groups = {Terrain.MOUNTAIN.value: 0, Terrain.GRASS.value: 1, Terrain.ROCK.value: 1, Terrain.RIVER.value: 2,
              Terrain.RIVER_STONE.value: 2}
    diff = [0, 0, 0]

    for y in range(rows):
        for x in range(columns):
            diff[groups[map1[y][x]]] += 1
            diff[groups[map2[y][x]]] -= 1

    return sum(abs(d) for d in diff)


def deterministic_crowding(parent1, parent2, child1, child2):
    if distance(parent1, child1) + distance(parent2, child2) <= distance(parent1, child2) + distance(parent2, child1):
        return [
            child1 if calc_fitness(child1) > calc_fitness(parent1) else parent1,
            child2 if calc_fitness(child2) > calc_fitness(parent2) else parent2
        ]
    else:
        return [
            child2 if calc_fitness(child2) > calc_fitness(parent1) else parent1,
            child1 if calc_fitness(child1) > calc_fitness(parent2) else parent2
        ]


def main():
    digit_maps = initialize_digit_maps()

    fitness_scores = [calc_fitness(digit_map) for digit_map in digit_maps]

    for _ in tqdm(range(1, generations + 1)):
        offspring = []
        while len(offspring) < population:
            parent1, parent2 = tournament_selection(digit_maps, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            mutation(child1)
            mutation(child2)
            child1 = apply_cellular_automata(child1)
            child2 = apply_cellular_automata(child2)
            child1 = replace_terrain(child1)
            child2 = replace_terrain(child2)
            offspring.extend(deterministic_crowding(parent1, parent2, child1, child2))
        digit_maps = offspring[:population]

    fitness_scores = [calc_fitness(digit_map) for digit_map in digit_maps]
    top_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:10]
    save_maps(digit_maps, top_indices)


if __name__ == '__main__':
    main()
