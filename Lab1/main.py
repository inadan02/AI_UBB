# from cmath import sqrt
import heapq
from math import sqrt
from queue import Queue
from heapq import heapify


def pb1(words):
    """
    O(n) (functia split are complexitatea de timp O(n)) n=nr de cuvinte
    :param words: string, textul dat
    :return: last: string, ultimul cuvant dpdv alfabetic
    """
    all_words = words.split()
    last = all_words[0]
    for current in all_words:
        if current > last:
            last = current
    return last


def pb1_v2(words):
    """
    O(nlogn)
    :param words: string, textul dat
    :return: string
    """
    all_words = words.split()
    all_words_desc = sorted(all_words, reverse=True)
    return all_words_desc[0]


def test_pb1():
    print(pb1("Ana are mere rosii si galbene"))
    print(pb1("zare zeita zaa"))

    print(pb1_v2("Ana are mere rosii si galbene"))
    print(pb1_v2("zare zeita zaa"))


def pb2(x1, y1, x2, y2):
    """
    O(1)
    :param x1: abscisa pentru primul punct
    :param y1: ordonata pentru primul punct
    :param x2: abscisa pentru al doilea punct
    :param y2: ordonata pentru al doilea punct
    :return:
    """
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))
    # return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def test_pb2():
    print(pb2(1, 5, 4, 1))


def pb3(v1, v2):
    """
    O(n)
    :param v1: primul vector
    :param v2: al doilea vector
    :return: rez: sprodusul scalar al celor doi vectori
    """
    l1 = len(v1)
    l2 = len(v2)
    if l1 != l2:
        return "dimensiuni diferite ale vectorilor"
    rez = 0
    for i in range(l1):
        rez += v1[i] * v2[i]
    return rez


def pb3_v2(v1, v2):
    """
    O(m), unde m este nr de elemente dif de 0 m<n, unde n este nr total de elemente
    :param v1: primul vector
    :param v2: al doilea vector
    :return: produsul scalar al celor doi vectori
    """
    d1 = {}
    l1 = len(v1)
    if l1 != len(v2):
        return "dimensiuni diferite ale vectorilor"
    for i in range(l1):
        if v1[i] != 0:
            d1[i] = v1[i]
    product = 0
    for i in d1:
        if v2[i] != 0:
            product += v2[i] * d1[i]
    return product


def test_pb3():
    print(pb3([1, 0, 2, 0, 3], [1, 2, 0, 3, 1]))
    print(pb3([1, 0, 2, 0, 3], [1, 2, 0, 3, 1, 5]))

    print(pb3_v2([1, 0, 2, 0, 3], [1, 2, 0, 3, 1]))
    print(pb3_v2([1, 0, 2, 0, 3], [1, 2, 0, 3, 1, 5]))


def pb4(words):
    """
    O(n)
    :param words: string, testul dat
    :return: ultimul cuvand dpdv alfabetic
    """
    all_words = set()
    repeats = set()
    for word in words.split():
        if word in all_words:
            repeats.add(word)  # O(1)
        all_words.add(word)  # O(1)
    return all_words - repeats


def test_pb4():
    print(pb4("ana are ana are mere rosii ana"))


def pb5(vect, n):
    """
    O(n), n=nr elemente vector
    :param vect: vectorul dat
    :param n: integer
    :return: valoare care se repeta
    """
    return sum(vect) - (((n - 1) * n) // 2)


def test_pb5():
    vect = [1, 2, 3, 4, 2]
    n = len(vect)
    print(pb5(vect, n))


def pb6(vect):
    """
    O(n)
    Boyer Moore
    :param vect: vectorul dat
    :return: elemetul majoritar daca exista sau false daca nu exista
    """
    maj_index = 0
    votes = 1
    l = len(vect)
    for i in range(l):
        if vect[maj_index] == vect[i]:
            votes += 1
        else:
            votes -= 1
        if votes == 0:  # nr egal de voturi pt diferite elemente, avem nevoide e elem majoritar deci acest lucuru nu este accepatbil=>alegem elementul curent pentru candidat
            maj_index = i
            votes = 1
    votes = 0
    for i in range(l):
        if vect[i] == vect[maj_index]:
            votes += 1
    if votes > len(vect) / 2:
        return vect[maj_index]
    else:
        return False


def test_pb6():
    print(pb6([2, 8, 7, 2, 2, 5, 2, 3, 1, 2, 2]))
    print(pb6([1, 2, 1, 3]))
    print(pb6([1, 2, 1, 3, 1]))


def pb7(lista, k):
    """
    O(nlogn)
    :param lista: vectorul dat
    :param k: integer, pozitia dorita a elem celui mai mare
    :return: al k-lea cel mai mare element
    """
    lista.sort()
    return lista[len(lista) - k]


def pb7_v2(lista, k):
    """
    min heap
    :param lista: vectorul dat
    :param k: integer, pozitia dorita a elem celui mai mare
    :return: al k-lea cel mai mare element
    """
    heapify(lista)
    return (heapq.nlargest(k, lista))[k - 1]


def test_pb7():
    lista = [7, 4, 6, 3, 9, 1]
    k = 2
    print(pb7(lista, k))
    print(pb7([10, 5, 6, 3, 8, 9], 3))

    print(pb7_v2(lista, k))


def pb8(n):
    """
    O(n)
    :param n: numarul dat
    :return: numerele bimare de la 1 la n
    """
    q = Queue()
    q.put("1")
    while n > 0:
        n -= 1
        s1 = q.get()
        print(s1)
        s2 = s1
        q.put(s1 + "0")
        q.put(s2 + "1")


def test_pb8():
    pb8(4)
    pb8(10)


def pb9(matr, perechi):
    """
    O(n*m)
    matricea incepe de la 0
    :param matr: matricea data
    :param perechi: lista de perechi ce reprezinta coordonatele a celor 2 casute
    :return:
    """
    rez = []
    for pereche1, pereche2 in perechi:
        suma = 0
        for i in range(pereche1[0], pereche2[0] + 1):
            for j in range(pereche1[1], pereche2[1] + 1):
                suma += matr[i][j]
        rez.append(suma)
    return rez


def test_pb9():
    print(pb9([[0, 2, 5, 4, 1],
               [4, 8, 2, 3, 7],
               [6, 3, 4, 6, 2],
               [7, 3, 1, 8, 3],
               [1, 5, 7, 9, 4]],
              [((1, 1), (3, 3)),
               ((2, 2), (4, 4))]))


def pb10(matr):
    """
    O(m+n)
    :param matr: matricea data
    :return: linia cu nr maxim de 1
    parcurgere de la dreapta la stanga
    """
    n = len(matr)
    m = len(matr[0])
    max_index_linie = 0
    col_curenta = m - 1
    for i in range(n):
        while col_curenta >= 0 and matr[i][col_curenta] == 1:
            col_curenta = col_curenta - 1
            max_index_linie = i
    return max_index_linie


def test_pb10():
    print(pb10([[0, 0, 0, 1, 1],
                [0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1]]))

    print(pb10([[0, 0, 0, 1, 1],
                [0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [1, 1, 1, 1, 1]]))


if __name__ == '__main__':
    # print("lista: ")
    # input_int_array = [int(x) for x in input().split()]
    # print(pb7(input_int_array, 2))
    print('--pb1--')
    test_pb1()
    print('--pb2--')
    test_pb2()
    print('--pb3--')
    test_pb3()
    print('--pb4--')
    test_pb4()
    print('--pb5--')
    test_pb5()
    print('--pb6--')
    test_pb6()
    print('--pb7--')
    test_pb7()
    print('--pb8--')
    test_pb8()
    print('--pb9--')
    test_pb9()
    print('--pb10--')
    test_pb10()
