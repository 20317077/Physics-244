import matplotlib.pyplot as plt
import numpy as np
import functions as f


def main():
    # data simulation
    start = 0.0
    stop = 4.0
    step_size = 0.04
    time_sample = np.arange(start, stop, step_size)
    number_of_steps = np.size(time_sample)

    # Supply odd number of arguments
    actual_parameters = [1.1, 0.3, 0.6, 0.8, -0.8]
    actual_function = sum(f.trig(time_sample, actual_parameters))

    # noise generation
    sigma = 0.5
    data = np.random.normal(actual_function, sigma, number_of_steps)

    # data fitting
    number_of_parameters = 11
    precision_normalised_data_z = data / sigma
    design_matrix_x = [value / sigma for value in f.trig(time_sample, np.ones(number_of_parameters))]
    hessian_h = design_matrix_x @ np.transpose(design_matrix_x)
    beta_hat_b = np.linalg.inv(hessian_h) @ design_matrix_x @ precision_normalised_data_z
    found_function = sum(f.trig(time_sample, beta_hat_b))

    # Plot
    fig, ax = plt.subplots()
    ax.plot(time_sample, actual_function, '--', label="Actual functionl")
    ax.plot(time_sample, data, '.', label="Experiment samples")
    ax.plot(time_sample, found_function, label="Found function")
    ax.grid()
    plt.title("Least square solution to gaussian noise distributed data")
    plt.ylabel("Data Value")
    plt.xlabel("Time")
    ax.legend()
    fig.savefig("test.png")
    plt.show()

    # 3.5 minimum chi_squared
    minimum_chi_squared_q = precision_normalised_data_z @ np.transpose(
        precision_normalised_data_z) - np.transpose(beta_hat_b) @ np.transpose(design_matrix_x) @ design_matrix_x @ beta_hat_b
    co = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)**number_of_steps
    maximum_likelihood = co * (np.e**(-0.5 * minimum_chi_squared_q))
    print(maximum_likelihood)


if __name__ == "__main__":
    main()
