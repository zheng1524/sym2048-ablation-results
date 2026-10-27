"""Numba bitboard kernels for complete enumeration and exact dynamic programming."""

import numpy as np
from numba import njit, prange

from game2048.rules import decode, permutations_d4


@njit(cache=True)
def _line_table(size):
    """Compact and merge packed rows; each input tile participates in at most one merge."""
    lines = np.zeros(1 << (4 * size), dtype=np.uint64)
    for code in range(len(lines)):
        pending = 0
        destination = 0
        merged = np.uint64(0)
        overflow = False
        for source in range(size):
            tile = (code >> (4 * source)) & 15
            if tile == 0:
                continue
            if pending == 0:
                pending = tile
            elif pending == tile:
                if tile == 15:
                    overflow = True
                    break
                merged |= np.uint64(tile + 1) << np.uint64(4 * destination)
                destination += 1
                pending = 0
            else:
                merged |= np.uint64(pending) << np.uint64(4 * destination)
                destination += 1
                pending = tile
        if not overflow:
            if pending:
                merged |= np.uint64(pending) << np.uint64(4 * destination)
            lines[code] = merged
    return lines


def make_tables(size: int) -> tuple[np.ndarray, np.ndarray]:
    if size not in (2, 3):
        raise ValueError("The exact solver supports sizes 2 and 3")
    width = 1 << (4 * size)
    lines = _line_table(size)
    sym = np.zeros((8, size, width), dtype=np.uint64)
    perms = permutations_d4(size)
    inverse = np.argsort(perms, axis=1)
    for code in range(width):
        row = decode(code, size).ravel()[:size]
        for r in range(size):
            for g in range(8):
                packed = 0
                for c in range(size):
                    packed |= int(row[c]) << (4 * int(inverse[g, r * size + c]))
                sym[g, r, code] = packed
    return lines, sym


@njit(cache=True)
def move_code(code, action, size, lines):
    out = np.uint64(0)
    for row in range(size):
        line = np.uint64(0)
        for col in range(size):
            c = size - 1 - col if action % 2 else col
            pos = c * size + row if action >= 2 else row * size + c
            line |= ((code >> np.uint64(4 * pos)) & np.uint64(15)) << np.uint64(4 * col)
        moved = lines[int(line)]
        for col in range(size):
            c = size - 1 - col if action % 2 else col
            pos = c * size + row if action >= 2 else row * size + c
            out |= ((moved >> np.uint64(4 * col)) & np.uint64(15)) << np.uint64(4 * pos)
    return out


@njit(cache=True)
def canonical_code(code, size, sym):
    best = np.uint64(0xFFFFFFFFFFFFFFFF)
    best_g = 0
    mask = np.uint64((1 << (4 * size)) - 1)
    for g in range(8):
        value = np.uint64(0)
        for row in range(size):
            chunk = int((code >> np.uint64(4 * size * row)) & mask)
            value |= sym[g, row, chunk]
        if value < best:
            best, best_g = value, g
    return best, best_g


@njit(cache=True)
def max_exponent(code, size):
    result = 0
    for i in range(size * size):
        result = max(result, int((code >> np.uint64(4 * i)) & np.uint64(15)))
    return result


@njit(cache=True)
def is_terminal(code, size, lines):
    for action in range(4):
        if move_code(code, action, size, lines) != code:
            return False
    return True


@njit(cache=True, parallel=True)
def successors(codes, size, lines, sym):
    """Bounded chunk output, split by spawned tile. Zero marks padding."""
    out = np.zeros((len(codes), 2, 4 * size * size), dtype=np.uint64)
    for i in prange(len(codes)):
        seen = np.zeros(4, dtype=np.uint64)
        count = 0
        for action in range(4):
            moved = move_code(codes[i], action, size, lines)
            if moved == codes[i]:
                continue
            canonical_after, _ = canonical_code(moved, size, sym)
            duplicate = False
            for j in range(count):
                if seen[j] == canonical_after:
                    duplicate = True
            if duplicate:
                continue
            seen[count] = canonical_after
            count += 1
            for pos in range(size * size):
                shift = np.uint64(4 * pos)
                if (moved >> shift) & np.uint64(15):
                    continue
                for tile in range(1, 3):
                    nxt = moved | (np.uint64(tile) << shift)
                    # Terminal values are analytic boundaries, not table entries.
                    if not is_terminal(nxt, size, lines):
                        value, _ = canonical_code(nxt, size, sym)
                        out[i, tile - 1, action * size * size + pos] = value
    return out


@njit(cache=True)
def find_sorted(codes, code):
    idx = np.searchsorted(codes, code)
    return idx if idx < len(codes) and codes[idx] == code else -1


@njit(cache=True, parallel=True)
def solve_layer(codes, next2, values2, next4, values4, size, lines, sym):
    q = np.zeros((len(codes), 4), dtype=np.float64)
    legal = np.zeros((len(codes), 4), dtype=np.bool_)
    missing = np.zeros(len(codes), dtype=np.int64)
    for i in prange(len(codes)):
        for action in range(4):
            moved = move_code(codes[i], action, size, lines)
            if moved == codes[i]:
                continue
            legal[i, action] = True
            total = 0.0
            empty = 0
            for pos in range(size * size):
                shift = np.uint64(4 * pos)
                if (moved >> shift) & np.uint64(15):
                    continue
                empty += 1
                for tile in range(1, 3):
                    nxt = moved | (np.uint64(tile) << shift)
                    if is_terminal(nxt, size, lines):
                        value = float(max_exponent(nxt, size))
                    else:
                        canonical_next, _ = canonical_code(nxt, size, sym)
                        keys = next2 if tile == 1 else next4
                        values = values2 if tile == 1 else values4
                        idx = find_sorted(keys, canonical_next)
                        if idx < 0:
                            missing[i] += 1
                            value = 0.0
                        else:
                            value = values[idx]
                    total += (0.9 if tile == 1 else 0.1) * value
            q[i, action] = total / empty
    return q, legal, missing


@njit(cache=True)
def hash_slot(code, mask):
    x = code
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return np.int64((x ^ (x >> np.uint64(31))) & np.uint64(mask))


@njit(cache=True)
def build_index(codes, slots):
    mask = len(slots) - 1
    for i in range(len(codes)):
        pos = hash_slot(codes[i], mask)
        while slots[pos] >= 0:
            pos = (pos + 1) & mask
        slots[pos] = i


@njit(cache=True, parallel=True)
def verify_index(codes, slots):
    failures = 0
    for i in prange(len(codes)):
        slot = hash_slot(codes[i], len(slots) - 1)
        while slots[slot] >= 0 and codes[slots[slot]] != codes[i]:
            slot = (slot + 1) & (len(slots) - 1)
        failures += slots[slot] != i
    return failures


@njit(cache=True, parallel=True)
def lookup_boards(boards, codes, slots, size, sym):
    rows = np.full(len(boards), -1, dtype=np.int64)
    groups = np.zeros(len(boards), dtype=np.int32)
    for i in prange(len(boards)):
        code = np.uint64(0)
        for pos in range(size * size):
            code |= np.uint64(boards[i, pos // size, pos % size]) << np.uint64(4 * pos)
        canon, group = canonical_code(code, size, sym)
        groups[i] = group
        slot = hash_slot(canon, len(slots) - 1)
        while slots[slot] >= 0:
            row = slots[slot]
            if codes[row] == canon:
                rows[i] = row
                break
            slot = (slot + 1) & (len(slots) - 1)
    return rows, groups
