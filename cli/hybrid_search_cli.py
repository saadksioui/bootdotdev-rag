import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command")

    normalize_parser = subparsers.add_parser(
        'normalize', help="Normalize the scores"
    )

    normalize_parser.add_argument(
        'nums', nargs='*', type=float
    )

    args = parser.parse_args()

    match args.command:
        case 'normalize':
            if not args.nums:
                return
            min_nums = min(args.nums)
            max_nums = max(args.nums)
            if min_nums == max_nums:
                for _ in range(len(args.nums)):
                    print(f"* 1.0000")
                return
            for score in args.nums:
                norm = (score - min_nums) / (max_nums - min_nums)
                print(f"* {norm:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()