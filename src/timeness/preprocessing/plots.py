def plot_correlation(dataset, x="Date", y="timeness"):
    dataset.plot(x=x, y=y, marker="o", linestyle="-", color="b")
    correlation = dataset[y].corr(dataset[x], method="kendall")
    print(f"Correlation: {correlation:.2f}")
