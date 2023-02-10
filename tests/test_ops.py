import pandas as pd

from aiq.ops import Rolling, Ref, Resi


if __name__ == '__main__':
    input = pd.Series([1, 2, 3])

    op = Rolling(2, 'mean')
    output = op(input)
    print(output)

    op = Ref(1)
    output = op(input)
    print(output)

    op = Resi(2)
    output = op(input)
    print(output)
