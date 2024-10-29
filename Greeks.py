import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class BlackScholesGreeks:
    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def calculate_greeks(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        delta = norm.cdf(d1)
        vega = self.S * norm.pdf(d1) * np.sqrt(self.T)
        theta = -(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(
            -self.r * self.T) * norm.cdf(d2)
        rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

        return delta, gamma, theta, vega, rho

    def calculate_option_price(self, S, K, T, r, sigma, option_type="call"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price

    def plot_greek_vs_variable(self, variable):
        variable_values = np.linspace(0.5 * getattr(self, variable), 1.5 * getattr(self, variable), 100)
        delta_vals, gamma_vals, theta_vals, vega_vals, rho_vals = [], [], [], [], []

        for val in variable_values:
            params = {'S': self.S, 'K': self.K, 'T': self.T, 'r': self.r, 'sigma': self.sigma}
            params[variable] = val
            greeks = self.__class__(**params)
            delta, gamma, theta, vega, rho = greeks.calculate_greeks()

            delta_vals.append(delta)
            gamma_vals.append(gamma)
            theta_vals.append(theta)
            vega_vals.append(vega)
            rho_vals.append(rho)

        fig, axs = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(f'Greek Sensitivity to {variable}', fontsize=16)

        greek_names = ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho']
        greek_vals = [delta_vals, gamma_vals, theta_vals, vega_vals, rho_vals]

        for ax, greek_name, vals in zip(axs.flat, greek_names, greek_vals):
            ax.plot(variable_values, vals, label=greek_name, color='b')
            ax.set_xlabel(variable)
            ax.set_ylabel(greek_name)
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()


def main():

    bs_greeks = BlackScholesGreeks(S=100, K=100, T=1, r=0.05, sigma=0.2)

    # Print out the calculated Greeks
    delta, gamma, theta, vega, rho = bs_greeks.calculate_greeks()
    print("Calculated Greeks:")
    print(f"Delta: {delta}")
    print(f"Gamma: {gamma}")
    print(f"Theta: {theta}")
    print(f"Vega: {vega}")
    print(f"Rho: {rho}")

    variable_to_plot = 'S'  # change this to any of 'S', 'K', 'T', 'r', or 'sigma'
    bs_greeks.plot_greek_vs_variable(variable_to_plot)


if __name__ == "__main__":
    main()
