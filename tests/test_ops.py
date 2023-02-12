import pandas as pd

from aiq.ops import Rolling, Ref, Resi, Log


if __name__ == '__main__':
    input = pd.Series([1, 2, 3])

    output = Ref(input, 2)
    print(output)

    output = Resi(input, 2)
    print(output)

    output = Log(input)
    print(output)
