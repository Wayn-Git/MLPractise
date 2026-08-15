import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


x = np.arange(1,101)
y = 3 * x + 6

print(x.size)

# plt.title("Checking linear relationship")
# plt.xlabel("Input Features")
# plt.ylabel("Output Features")
# sns.regplot(x=x, y=y, ci=None, line_kws={"color": "red"})
# plt.show()

class Linear():
    def __init__(self, in_feature, out_feature, bias=True):
        self.weights = np.random.randn(out_feature, in_feature)

        print(self.weights)

        if bias == True:
            self.bias = np.random.randn(out_feature)
            print(self.bias)
        else:
            self.bias = None

    def forward(self, x):
        output = x @ self.weights.T

        print(output)

        if self.bias is not None:
            output += self.bias
        return output



L = Linear(100, 0)


y = L.forward(x)
print(y)
