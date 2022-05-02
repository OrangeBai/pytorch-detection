def test(a, b, c=0, *args, **kwargs):
    print(a)
    print(b)
    print(c)


if __name__ == '__main__':
    blabla = {'c': {10}}

    test(1, 2, **blabla)

    print(1)
