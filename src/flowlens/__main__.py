if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    from flowlens.app import main

    raise SystemExit(main())
