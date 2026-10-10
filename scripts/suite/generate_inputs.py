"""
Realized by @luckignolo32 (GitHub) - Creative Commons Attribution-ShareAlike 4.0 International

Deterministic generator for the storm input files that the multicore test
suite needs but that are not shipped with the original handout. Every file is
produced from a fixed seed, so re-running this script always regenerates
byte-identical inputs. The files are written to DuoWaves/custom-tests and are
meant to be committed to the DuoWaves repository so that the cluster clones
receive them through a normal git pull.
"""

import pathlib
import random
import sys

POSITION_VALUE_LOW = 50
POSITION_VALUE_HIGH = 149


def duowaves_dir():
    """
    TR-4
    Resolves the DuoWaves repository as a sibling of the multicore-relation
    repository that contains this script, so the generator can be invoked
    from any working directory both locally and on the cluster.
    """
    suite = pathlib.Path(__file__).resolve().parent
    return suite.parent.parent.parent / "DuoWaves"


def write_storm_file(path, particles, max_position, seed):
    """
    TM-B, TM-C
    Writes one storm file in the handout format: the first line holds the
    particle count and every following line holds one particle as an integer
    position followed by an integer energy value. Positions are drawn
    uniformly in [0, max_position] and values uniformly in [50, 149], which
    matches the ranges observed in the original test_02 wave files.
    """
    rng = random.Random(seed)
    lines = [str(particles)]
    for _ in range(particles):
        lines.append(f"{rng.randint(0, max_position)} {rng.randint(POSITION_VALUE_LOW, POSITION_VALUE_HIGH)}")
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path} ({particles} particles, seed {seed})")


def main():
    """
    TM-B, TM-C
    Generates the four extra test_02 waves (w7 to w10) required by the storm
    count scaling test, which needs ten cumulative storm files while the
    handout only ships six, and the four single-wave files with 3, 100, 1000
    and 5000 particles required by the particle count scaling test at a fixed
    layer of 20000 positions.
    """
    target = duowaves_dir() / "custom-tests"
    if not target.parent.is_dir():
        sys.exit(f"DuoWaves repository not found at {target.parent}")
    target.mkdir(exist_ok=True)

    for wave in range(7, 11):
        write_storm_file(target / f"test_02_a30k_p20k_w{wave}", 20000, 29999, 3000 + wave)

    particle_files = [
        ("test_10_a20k_p3_w1", 3, 4010),
        ("test_11_a20k_p100_w1", 100, 4011),
        ("test_12_a20k_p1k_w1", 1000, 4012),
        ("test_13_a20k_p5k_w1", 5000, 4013),
    ]
    for name, particles, seed in particle_files:
        write_storm_file(target / name, particles, 19999, seed)


if __name__ == "__main__":
    main()
