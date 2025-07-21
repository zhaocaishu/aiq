import pandas as pd

from aiq.ops import Ref, Resi, Log, Rank


if __name__ == "__main__":
    input = pd.Series([1, 2, 3, 4])

    output = Ref(input, 2)
    print(output)

    output = Resi(input, 2)
    print(output)

    output = Log(input)
    print(output)

    output = Rank(input, 2)
    print(output)
