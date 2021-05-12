def main():
    mini_sample = [(1, 'a', "AA"), (2, 'b', "BB")]

    a, b, c = [list(seq) for seq in zip(*mini_sample)]
    print()


if __name__ == '__main__':
    main()
