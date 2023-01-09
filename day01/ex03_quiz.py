import torch


def quiz1():

    A = torch.arange(1, 7).view(2, 3)
    B = torch.arange(10, 70, 10).view(2, 3)

    addition = A + B
    substraction = A - B
    sumA = A.sum()
    sumB = B.sum()

    print(f'Addition A and B \n\t{addition}')
    print(f'Substraction A and B \n\t{substraction}')
    print(f'Sum of All Elements of A \n\t{sumA}')
    print(f'Sum of All Elements of B \n\t{sumB}')

    return (
        addition,
        substraction,
        sumA,
        sumB,
    )


def quiz2():
    arr = torch.arange(1, 46).view(1, 5, 3, 3)
    arr = torch.transpose(a, 1, 3)
    print(arr[0, 2, 2, :])


def quiz3():
    A = torch.arange(1, 7).view(2, 3)
    B = torch.arange(10, 70, 10).view(2, 3)
    AB_cat = torch.cat([A, B], dim=1)
    AB_st = torch.stack([A, B], dim=0)
    print(AB_cat)
    print(AB_st)


if __name__ == '__main__':
    quiz3()
